from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexicon_vi_fin import FEATURE_NAMES, score_text  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "interim" / "to_label_v2.parquet"
SEED_FILE = ROOT / "data" / "interim" / "claude_seed_labels.csv"
GOLD_FILE = ROOT / "data" / "interim" / "gold_seed_v2.csv"
OUT_FILE = ROOT / "data" / "interim" / "labeled_auto.jsonl"
REPORT = ROOT / "docs" / "prelabel_local_report.md"

SEED = 42
N_FOLDS = 5
SELF_TRAIN_TH = 0.80
MANUAL_CONF = 0.95
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def build_features(texts: list[str], vecs=None, scaler=None, fit: bool = False):
    if fit:
        vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             max_features=60000, sublinear_tf=True)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                             max_features=80000, sublinear_tf=True)
        Xw, Xc = vw.fit_transform(texts), vc.fit_transform(texts)
        vecs = (vw, vc)
    else:
        vw, vc = vecs
        Xw, Xc = vw.transform(texts), vc.transform(texts)

    lex = np.array([[score_text(t)[k] for k in FEATURE_NAMES] for t in texts],
                   dtype=np.float64)
    if fit:
        scaler = StandardScaler().fit(lex)
    Xl = csr_matrix(scaler.transform(lex))

    return hstack([Xw, Xc, Xl]).tocsr(), vecs, scaler


def make_clf() -> CalibratedClassifierCV:
    base = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced",
                              random_state=SEED)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def reliability_table(conf: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    idx = np.digitize(conf, bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "conf range": f"{bins[b]:.2f}-{bins[b+1]:.2f}",
            "n": int(m.sum()),
            "avg conf": round(float(conf[m].mean()), 3),
            "actual accuracy": round(float(correct[m].mean()), 3),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="delete existing jsonl and relabel from scratch")
    ap.add_argument("--report", action="store_true", help="print stats for existing file only")
    args = ap.parse_args()

    if args.report:
        return summarize()

    print("=" * 68)
    print("PRELABEL LOCAL - no API calls, no cost")
    print("=" * 68)

    df = pd.read_parquet(IN_FILE)
    df["id"] = df["id"].astype(str)
    print(f"  to_label_v2      : {len(df):,} articles")

    if not SEED_FILE.exists():
        print(f"  MISSING: {SEED_FILE}. Cannot continue.")
        return 1
    seed = pd.read_csv(SEED_FILE, encoding="utf-8-sig")
    seed["id"] = seed["id"].astype(str)
    seed = seed[seed["label"].isin(LABELS)].drop_duplicates(subset="id")
    print(f"  seed labels      : {len(seed):,} articles ({len(seed)/len(df):.1%})")

    gold_ids: set[str] = set()
    if GOLD_FILE.exists():
        g = pd.read_csv(GOLD_FILE, encoding="utf-8-sig")
        g = g[g["label"].notna() & (g["label"].astype(str).str.strip() != "")]
        gold_ids = set(g["id"].astype(str))
    print(f"  human gold set   : {len(gold_ids):,} articles - EXCLUDED from training")

    if args.rebuild and OUT_FILE.exists():
        OUT_FILE.unlink()

    done_ids: set[str] = set()
    if OUT_FILE.exists():
        with OUT_FILE.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(str(json.loads(line)["id"]))
                    except Exception:
                        pass
        print(f"  resume           : skipping {len(done_ids):,} already labeled")

    text_by_id = dict(zip(df["id"], df["text_input"].astype(str)))

    train = seed.copy()
    train["text"] = train["id"].map(text_by_id)
    train = train.dropna(subset=["text"])
    print(f"  train set        : {len(train):,} articles")
    print("  distribution:", dict(Counter(train["label"])))

    Xtr, vecs, scaler = build_features(train["text"].tolist(), fit=True)
    ytr = train["label"].to_numpy()

    print("\n" + "-" * 68)
    print("Cross-validation 5-fold on seed labels")
    print("-" * 68)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    proba_cv = cross_val_predict(make_clf(), Xtr, ytr, cv=cv, method="predict_proba")
    classes = np.array(sorted(set(ytr)))
    pred_cv = classes[proba_cv.argmax(axis=1)]
    conf_cv = proba_cv.max(axis=1)
    correct = (pred_cv == ytr).astype(float)

    acc = accuracy_score(ytr, pred_cv)
    mf1 = f1_score(ytr, pred_cv, average="macro")
    print(f"  accuracy  : {acc:.4f}")
    print(f"  macro-F1  : {mf1:.4f}")
    print(classification_report(ytr, pred_cv, digits=3, zero_division=0))
    rel = reliability_table(conf_cv, correct)
    print("Confidence reliability:")
    print(rel.to_string(index=False))

    clf = make_clf().fit(Xtr, ytr)
    rest = df[~df["id"].isin(set(train["id"]))].reset_index(drop=True)
    Xre, _, _ = build_features(rest["text_input"].astype(str).tolist(), vecs, scaler)
    p1 = clf.predict_proba(Xre)

    conf1 = p1.max(axis=1)
    lab1 = classes[p1.argmax(axis=1)]
    take = conf1 >= SELF_TRAIN_TH
    print(f"\nSelf-training: adding {int(take.sum()):,} predictions with conf >= {SELF_TRAIN_TH}")

    aug_texts = train["text"].tolist() + rest.loc[take, "text_input"].astype(str).tolist()
    aug_y = np.concatenate([ytr, lab1[take]])
    Xa, vecs2, scaler2 = build_features(aug_texts, fit=True)
    clf2 = make_clf().fit(Xa, aug_y)

    Xre2, _, _ = build_features(rest["text_input"].astype(str).tolist(), vecs2, scaler2)
    p2 = clf2.predict_proba(Xre2)
    conf2 = p2.max(axis=1)
    lab2 = classes[p2.argmax(axis=1)]
    print(f"  avg confidence after self-training: {conf2.mean():.3f} "
          f"(before: {conf1.mean():.3f})")

    manual = dict(zip(train["id"], train["label"]))
    pred_map = dict(zip(rest["id"], zip(lab2, conf2)))

    n_new = 0
    with OUT_FILE.open("a", encoding="utf-8") as out:
        for _, row in df.iterrows():
            rid = row["id"]
            if rid in done_ids:
                continue
            if rid in manual:
                label, conf, src = manual[rid], MANUAL_CONF, "claude_manual"
                reason = "hand-labeled following docs/labeling_guide.md"
            else:
                label, conf = pred_map[rid]
                src = "weak_model"
                reason = "label propagation from seed set (tfidf+lexicon+LR)"
            out.write(json.dumps({
                "id": rid,
                "primary_ticker": str(row.get("primary_ticker", "")),
                "label": label,
                "confidence": round(float(conf), 4),
                "reason": reason,
                "label_source": src,
                "title": str(row.get("title", ""))[:200],
                "date": str(row.get("date", "")),
            }, ensure_ascii=False) + "\n")
            n_new += 1

    print(f"\nWrote {n_new:,} rows to {OUT_FILE.name}")
    write_report(len(df), len(train), acc, mf1, rel, classes, ytr, pred_cv, gold_ids)
    return summarize()


def write_report(n_total, n_seed, acc, mf1, rel, classes, ytr, pred_cv, gold_ids) -> None:
    cm = pd.crosstab(pd.Series(ytr, name="Hand-labeled"),
                     pd.Series(pred_cv, name="Model prediction"))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as f:
        w = f.write
        w("# Prelabel local report\n\n")
        w("No API calls, no cost. Label propagation from seed labels.\n\n")
        w("## 1. Scale\n\n")
        w(f"- Total articles to label: **{n_total:,}**\n")
        w(f"- Seed labels: **{n_seed:,}** ({n_seed/n_total:.1%})\n")
        w(f"- Human gold set (excluded from training): {len(gold_ids):,}\n\n")
        w("## 2. Model quality (5-fold CV on seed labels)\n\n")
        w(f"- accuracy: **{acc:.4f}**\n- macro-F1: **{mf1:.4f}**\n\n")
        w("### Confusion matrix\n\n")
        w(cm.to_markdown() + "\n\n")
        w("### Confidence reliability\n\n")
        w(rel.to_markdown(index=False) + "\n\n")
        w("The `actual accuracy` column shows what fraction of predictions in that confidence "
          "band are correct. Use this table to choose the confidence threshold in "
          "`finalize_dataset.py`.\n\n")
        w("## 3. label_source values in labeled_auto.jsonl\n\n")
        w("| Value | Meaning | confidence |\n|---|---|---|\n")
        w("| `claude_manual` | AI assistant read and labeled each article following the guide | 0.95 |\n")
        w("| `weak_model` | Label propagation prediction | calibrated probability |\n\n")
        w("## 4. Limitations\n\n")
        w("- Seed labels come from a language model, not a financial expert.\n")
        w("- Propagation model reads only headline and lead, not full text.\n")
        w("- Seed labels are biased toward the later part of the date range, so train "
          "articles (earlier dates) are mostly `weak_model` with higher noise.\n")
        w("- `weak_model` labels are weak labels. All model quality reports must be "
          "reported separately on `label_source == \"human\"` rows.\n")


def summarize() -> int:
    if not OUT_FILE.exists():
        print("No labeled_auto.jsonl yet")
        return 1
    recs = []
    with OUT_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    total = len(recs)
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"labeled_auto.jsonl: {total:,} rows")
    cnt = Counter(r["label"] for r in recs)
    for lbl in LABELS:
        pct = cnt.get(lbl, 0) / max(total, 1) * 100
        print(f"  {lbl:9s} {cnt.get(lbl,0):6,}  ({pct:5.1f}%)  {'#' * int(pct/2)}")
    src = Counter(r.get("label_source", "?") for r in recs)
    print("\nLabel sources:", dict(src))
    conf = np.array([r["confidence"] for r in recs])
    print(f"confidence: avg={conf.mean():.3f}  <0.5: {int((conf<0.5).sum()):,}")

    ok = True
    if total < 5000:
        print(f"FAIL: only {total:,} rows")
        ok = False
    for lbl in LABELS:
        pct = cnt.get(lbl, 0) / max(total, 1) * 100
        if pct < 5:
            print(f"FAIL: label {lbl} is {pct:.1f}% (<5%)")
            ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
