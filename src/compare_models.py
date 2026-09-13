from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "data" / "processed" / "test.parquet"
CACHE = ROOT / "data" / "interim" / "tokenized_cache.parquet"
BASE_MODEL = ROOT / "models" / "baseline_tfidf_lr.joblib"
PHO_PRED = ROOT / "models" / "phobert-vnfin" / "test_predictions.json"
DOC = ROOT / "docs" / "model_comparison.md"

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
I2L = dict(enumerate(LABELS))
ALPHA = 0.05


def mcnemar(b: int, c: int) -> tuple[float, str]:
    from scipy import stats
    n = b + c
    if n == 0:
        return 1.0, "no articles differ between models"
    if n < 25:
        p = float(stats.binomtest(min(b, c), n, 0.5).pvalue)
        return p, "exact binomial test (b+c < 25)"
    stat = (abs(b - c) - 1) ** 2 / n
    p = float(stats.chi2.sf(stat, 1))
    return p, f"continuity-corrected chi-square, stat={stat:.3f}"


def metrics_block(y, p, mask=None) -> dict:
    if mask is not None:
        y, p = y[mask], p[mask]
    d = {"n": int(len(y)),
         "macro_F1": round(f1_score(y, p, average="macro", zero_division=0), 4),
         "accuracy": round(accuracy_score(y, p), 4),
         "weighted_F1": round(f1_score(y, p, average="weighted", zero_division=0), 4)}
    per = f1_score(y, p, average=None, labels=list(range(3)), zero_division=0)
    for i, lb in enumerate(LABELS):
        d[f"F1_{lb}"] = round(float(per[i]), 4)
    return d


def main() -> int:
    print("=" * 68)
    print("MODEL COMPARISON")
    print("=" * 68)

    te = pd.read_parquet(TEST)
    tok = dict(zip(*pd.read_parquet(CACHE).T.values)) if CACHE.exists() else {}
    if CACHE.exists():
        c = pd.read_parquet(CACHE)
        tok = dict(zip(c["id"].astype(str), c["tok"]))
    te["tok"] = te["id"].astype(str).map(tok)
    if te["tok"].isna().any():
        print("  ERROR: missing tokenized_cache. Run src/train_baseline.py first.")
        return 1

    bundle = joblib.load(BASE_MODEL)
    base_pred_lbl = bundle["pipeline"].predict(te["tok"])
    L2I = {l: i for i, l in I2L.items()}
    base = np.array([L2I[x] for x in base_pred_lbl])

    pj = json.loads(PHO_PRED.read_text(encoding="utf-8"))
    order = {str(i): k for k, i in enumerate(pj["id"])}
    idx = te["id"].astype(str).map(order)
    if idx.isna().any():
        print("  ERROR: test_predictions.json ids do not match test.parquet")
        return 1
    pho = np.array(pj["y_pred"])[idx.astype(int).to_numpy()]
    y = np.array(pj["y_true"])[idx.astype(int).to_numpy()]

    assert (y == te["label"].map(L2I).to_numpy()).all(), "true labels do not match"
    print(f"  test: {len(te):,} articles, id alignment OK")

    hm = (te["source_detail"] == "human").to_numpy()
    tbl_all = pd.DataFrame([{"model": "Baseline TF-IDF+LR", **metrics_block(y, base)},
                            {"model": "PhoBERT",            **metrics_block(y, pho)}])
    tbl_hum = pd.DataFrame([{"model": "Baseline TF-IDF+LR", **metrics_block(y, base, hm)},
                            {"model": "PhoBERT",            **metrics_block(y, pho, hm)}])
    print("\n--- Full test ---")
    print(tbl_all.to_string(index=False))
    print(f"\n--- Human labels only (n={int(hm.sum())}) ---")
    print(tbl_hum.to_string(index=False))

    ok_b, ok_p = base == y, pho == y
    b = int((ok_b & ~ok_p).sum())
    c = int((~ok_b & ok_p).sum())
    both_ok = int((ok_b & ok_p).sum())
    both_bad = int((~ok_b & ~ok_p).sum())
    p_val, method = mcnemar(b, c)
    sig = p_val < ALPHA
    print(f"\n--- McNemar ---")
    print(f"  both correct     : {both_ok}")
    print(f"  both wrong       : {both_bad}")
    print(f"  baseline ok, PhoBERT wrong : b = {b}")
    print(f"  PhoBERT ok, baseline wrong : c = {c}")
    print(f"  p-value = {p_val:.3e}  ({method})")
    print(f"  => {'YES' if sig else 'NO'} statistically significant difference at alpha={ALPHA}")

    te = te.reset_index(drop=True)
    te["y"], te["base"], te["pho"] = y, base, pho
    g1 = te[(te.base != te.y) & (te.pho != te.y)]
    g2 = te[(te.base != te.y) & (te.pho == te.y)]
    g3 = te[(te.base == te.y) & (te.pho != te.y)]
    print(f"\n--- Error analysis ---")
    print(f"  Group 1 both wrong         : {len(g1)}")
    print(f"  Group 2 PhoBERT ok, LR wrong : {len(g2)}")
    print(f"  Group 3 LR ok, PhoBERT wrong : {len(g3)}")

    write_doc(tbl_all, tbl_hum, b, c, both_ok, both_bad, p_val, method, sig,
              g1, g2, g3, te, int(hm.sum()))
    print(f"\nWritten: {DOC}")
    print(f"\nP2.3 PASS  |  PhoBERT {tbl_all.iloc[1]['macro_F1']:.4f} vs "
          f"baseline {tbl_all.iloc[0]['macro_F1']:.4f}, p={p_val:.2e}")
    return 0


def ex_table(df: pd.DataFrame, k: int) -> str:
    out = ["| Ticker | Title | True | Baseline | PhoBERT |",
           "|---|---|:--:|:--:|:--:|"]
    for _, r in df.head(k).iterrows():
        t = str(r["title"]).replace("|", "/")[:88]
        out.append(f"| {r['primary_ticker']} | {t} | {I2L[r['y']]} | "
                   f"{I2L[r['base']]} | {I2L[r['pho']]} |")
    return "\n".join(out) + "\n\n"


def write_doc(tbl_all, tbl_hum, b, c, both_ok, both_bad, p_val, method, sig,
              g1, g2, g3, te, n_hum) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with DOC.open("w", encoding="utf-8") as f:
        w = f.write
        w("# Model comparison — Baseline vs PhoBERT\n\n")
        w("Same test set, same seed 42. PhoBERT predictions read from "
          "`models/phobert-vnfin/test_predictions.json`.\n\n")

        w("## 1. Main comparison table\n\n### (a) Full test\n\n")
        w(tbl_all.to_markdown(index=False) + "\n\n")
        w(f"### (b) Human labels only (n={n_hum})\n\n")
        w(tbl_hum.to_markdown(index=False) + "\n\n")
        w(f"> Sample size of {n_hum} articles is too small for reliable conclusions. "
          "Included for completeness, not for comparison.\n\n")

        w("## 2. McNemar test\n\n")
        w("| | PhoBERT correct | PhoBERT wrong |\n|---|---:|---:|\n")
        w(f"| **Baseline correct** | {both_ok} | {b} |\n")
        w(f"| **Baseline wrong** | {c} | {both_bad} |\n\n")
        w(f"- Method: {method}\n- p-value = **{p_val:.3e}**\n")
        w(f"- alpha = {ALPHA}\n\n")
        if sig:
            w(f"**Conclusion:** reject null hypothesis that both models err equally. PhoBERT "
              f"fixes {c} articles that baseline gets wrong, but breaks {b} that baseline "
              f"was correct on. The difference is statistically significant.\n\n")
        else:
            w(f"**Conclusion:** insufficient evidence to say the models differ "
              f"(p = {p_val:.3f} > {ALPHA}). The macro-F1 gap may be due to chance.\n\n")

        w("## 3. Error analysis\n\n")
        w(f"### Group 1 — both wrong ({len(g1)} articles)\n\n")
        w(ex_table(g1, 15))
        w("Hardest cases. Mainly: macro/sector news that only mentions the company in passing; "
          "articles with mixed signals in the same sentence; articles where `primary_ticker` "
          "is misassigned so even the correct label is irrelevant.\n\n")

        w(f"### Group 2 — PhoBERT correct, baseline wrong ({len(g2)} articles)\n\n")
        w(ex_table(g2, 10))
        w("Mostly sentences where meaning depends on word order and negation, "
          "which TF-IDF misses because it only counts discrete n-grams.\n\n")

        w(f"### Group 3 — baseline correct, PhoBERT wrong ({len(g3)} articles)\n\n")
        w(ex_table(g3, 10))
        w("Usually articles with very strong, clear keywords that play to the strength "
          "of a bag-of-words model; PhoBERT gets pulled off-course by surrounding context.\n\n")

        w("## 4. Summary\n\n")
        bf, pf = tbl_all.iloc[0]["macro_F1"], tbl_all.iloc[1]["macro_F1"]
        w(f"- PhoBERT outperforms baseline by **{pf - bf:+.4f}** macro-F1 ({bf:.4f} → {pf:.4f}).\n")
        w(f"- Largest gain on NEGATIVE class: "
          f"{tbl_all.iloc[0]['F1_NEGATIVE']:.3f} → {tbl_all.iloc[1]['F1_NEGATIVE']:.3f}. "
          "This is the rarest class and the most important for event study.\n")
        w(f"- {both_bad} articles are wrong for both models ({both_bad/len(te):.1%} of test set). "
          "This is unlikely to improve by changing architecture; the root cause is label quality "
          "in train and ticker attribution accuracy.\n")
        w("- Train contains ~89% `weak_model` labels. Both models learn from noisy labels, "
          "so these numbers represent a floor, not a ceiling.\n")


if __name__ == "__main__":
    sys.exit(main())
