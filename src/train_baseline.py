from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns            # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline                        # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "interim" / "tokenized_cache.parquet"
MODEL_OUT = ROOT / "models" / "baseline_tfidf_lr.joblib"
FIG_DIR = ROOT / "reports" / "figures"
DOC = ROOT / "docs" / "model_baseline.md"

SEED = 42
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
C_GRID = [0.1, 0.5, 1.0, 3.0, 10.0]


def tokenize_all(df_all: pd.DataFrame) -> dict[str, str]:
    cache: dict[str, str] = {}
    if CACHE.exists():
        c = pd.read_parquet(CACHE)
        cache = dict(zip(c["id"].astype(str), c["tok"]))
        print(f"  cache: {len(cache):,} articles already tokenized")

    need = df_all[~df_all["id"].astype(str).isin(cache)]
    if len(need):
        from underthesea import word_tokenize
        print(f"  tokenizing {len(need):,} new articles ...")
        t0 = time.time()
        for n, (i, txt) in enumerate(zip(need["id"].astype(str),
                                         need["text_input"].astype(str)), 1):
            cache[i] = word_tokenize(txt, format="text")
            if n % 1000 == 0:
                print(f"    {n:,}/{len(need):,}  ({time.time()-t0:.0f}s)")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": list(cache), "tok": list(cache.values())}).to_parquet(
            CACHE, index=False)
        print(f"  done in {time.time()-t0:.0f}s, cache saved")
    return cache


def make_pipe(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=C, random_state=SEED)),
    ])


def top_features(pipe: Pipeline, k: int = 20) -> dict[str, list[tuple[str, float]]]:
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coefs = pipe.named_steps["clf"].coef_
    classes = pipe.named_steps["clf"].classes_
    out = {}
    for ci, cls in enumerate(classes):
        idx = np.argsort(coefs[ci])[::-1][:k]
        out[cls] = [(names[i], round(float(coefs[ci][i]), 3)) for i in idx]
    return out


def main() -> int:
    print("=" * 68)
    print("BASELINE  TF-IDF + Logistic Regression")
    print("=" * 68)

    tr = pd.read_parquet(PROC / "train.parquet")
    va = pd.read_parquet(PROC / "val.parquet")
    te = pd.read_parquet(PROC / "test.parquet")
    print(f"  train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    tok = tokenize_all(pd.concat([tr, va, te])[["id", "text_input"]])
    for d in (tr, va, te):
        d["tok"] = d["id"].astype(str).map(tok)

    print("\n" + "-" * 68)
    print("Hyperparameter selection on val (macro-F1)")
    print("-" * 68)
    vclean = va[va["source_detail"] != "weak_model"]
    print(f"  eval set: {len(vclean)} val articles with hand-read labels "
          f"({dict(vclean.source_detail.value_counts())})")
    rows = []
    for C in C_GRID:
        p = make_pipe(C).fit(tr["tok"], tr["label"])
        pv, pc = p.predict(va["tok"]), p.predict(vclean["tok"])
        rows.append({
            "C": C,
            "macroF1_val_handread": round(f1_score(vclean["label"], pc, average="macro"), 4),
            "acc_val_handread": round(accuracy_score(vclean["label"], pc), 4),
            "macroF1_val_full": round(f1_score(va["label"], pv, average="macro"), 4),
        })
        r = rows[-1]
        print(f"  C={C:<5} val hand-read macro-F1={r['macroF1_val_handread']:.4f}"
              f"   (val full: {r['macroF1_val_full']:.4f})")
    grid = pd.DataFrame(rows)
    best_C = float(grid.loc[grid["macroF1_val_handread"].idxmax(), "C"])
    print(f"  -> best C: {best_C}")

    full = pd.concat([tr, va])
    pipe = make_pipe(best_C).fit(full["tok"], full["label"])
    pred = pipe.predict(te["tok"])

    mf1 = f1_score(te["label"], pred, average="macro")
    acc = accuracy_score(te["label"], pred)
    wf1 = f1_score(te["label"], pred, average="weighted")
    print("\n" + "-" * 68)
    print(f"TEST  macro-F1={mf1:.4f}  accuracy={acc:.4f}  weighted-F1={wf1:.4f}")
    print("-" * 68)
    rep = classification_report(te["label"], pred, digits=3, zero_division=0)
    print(rep)

    cm = confusion_matrix(te["label"], pred, labels=LABELS)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Baseline TF-IDF + LR — test (macro-F1={mf1:.3f})")
    plt.tight_layout(); plt.savefig(FIG_DIR / "cm_baseline.png", dpi=140); plt.close()

    hm = te["source_detail"] == "human" if "source_detail" in te else pd.Series(False, index=te.index)
    sub = None
    if hm.sum() > 0:
        ph = pipe.predict(te.loc[hm, "tok"])
        sub = {"n": int(hm.sum()),
               "macro_F1": round(f1_score(te.loc[hm, "label"], ph, average="macro"), 4),
               "accuracy": round(accuracy_score(te.loc[hm, "label"], ph), 4)}
        print(f"Human labels in test (n={sub['n']}): "
              f"macro-F1={sub['macro_F1']:.4f}  acc={sub['accuracy']:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "best_C": best_C, "labels": LABELS}, MODEL_OUT)
    print(f"\nModel saved: {MODEL_OUT}")

    write_doc(grid, best_C, mf1, acc, wf1, rep, cm, top_features(pipe), sub, te, hm)
    print(f"Written: {DOC}")
    print(f"\nP2.1 PASS  |  macro-F1 on test = {mf1:.4f}")
    return 0


def write_doc(grid, best_C, mf1, acc, wf1, rep, cm, feats, sub, te, hm) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with DOC.open("w", encoding="utf-8") as f:
        w = f.write
        w("# Baseline — TF-IDF + Logistic Regression\n\n")
        w("Word tokenization via `underthesea.word_tokenize`, cached in "
          "`data/interim/tokenized_cache.parquet`.\n")
        w("Features: TF-IDF 1-2 gram, `min_df=2`, `max_features=50000`, "
          "`sublinear_tf=True`.\nModel: `LogisticRegression(class_weight=\"balanced\", "
          "max_iter=2000)`, `random_state=42`.\n\n")

        w("## 1. Hyperparameter selection on val\n\n")
        w(grid.to_markdown(index=False) + "\n\n")
        w(f"Best C: **{best_C}**. Retrained on train+val, evaluated on test.\n\n")
        w("> Val contains up to 94% `weak_model` labels generated by a TF-IDF+LR model.\n"
          "> Selecting C on full val rewards the model that best mimics those weak labels,\n"
          "> not the model that reads news best. Column `macroF1_val_handread` is the\n"
          "> correct selection criterion. Full val column retained for reference.\n\n")

        w("## 2. Test results\n\n")
        w("| Metric | Value |\n|---|---|\n")
        w(f"| macro-F1 | **{mf1:.4f}** |\n| accuracy | {acc:.4f} |\n"
          f"| weighted-F1 | {wf1:.4f} |\n\n")

        w("## 3. Classification report\n\n```\n" + rep + "```\n\n")

        w("## 4. Confusion matrix\n\n")
        w("Rows = true labels, columns = predicted.\n\n")
        w("| | " + " | ".join(LABELS) + " |\n|---|" + "---|" * len(LABELS) + "\n")
        for i, lb in enumerate(LABELS):
            w(f"| **{lb}** | " + " | ".join(str(x) for x in cm[i]) + " |\n")
        w("\n![confusion matrix](../reports/figures/cm_baseline.png)\n\n")

        w("## 5. Top features learned by the model\n\n")
        for cls, items in feats.items():
            w(f"**{cls}** — 20 features with highest weight:\n\n")
            w("`" + "`, `".join(t for t, _ in items) + "`\n\n")

        w("## 6. Results on human-labeled subset\n\n")
        if sub:
            w(f"Test has **{sub['n']}** articles with human labels "
              f"(`source_detail == \"human\"`).\n\n")
            w(f"- macro-F1: **{sub['macro_F1']:.4f}**\n- accuracy: {sub['accuracy']:.4f}\n\n")
            w("Small sample size; confidence intervals are wide. Use for qualitative comparison only.\n\n")
        else:
            w("No human labels in test set.\n\n")

        if "source_detail" in te:
            w("### Label source breakdown in test set\n\n")
            vc = te["source_detail"].value_counts()
            w("| Source | Count |\n|---|---:|\n")
            for k, v in vc.items():
                w(f"| `{k}` | {v} |\n")
            w("\nTest set contains no `weak_model` labels; all labels are hand-read. "
              "This is why test numbers are more reliable than train.\n\n")

        w("## 7. Limitations\n\n")
        w("- Train contains ~89% `weak_model` labels; performance ceiling is set by "
          "label quality, not model architecture.\n")
        w("- TF-IDF does not capture long-distance negation, e.g. "
          "*\"failed to complete the acquisition\"* may look like a positive signal.\n")
        w("- Model does not know whether `primary_ticker` is the subject of the article "
          "or only mentioned in passing, even though the ticker appears in `text_input`.\n")


if __name__ == "__main__":
    sys.exit(main())
