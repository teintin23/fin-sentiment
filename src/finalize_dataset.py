from __future__ import annotations

import io
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLD_CSV = ROOT / "data" / "interim" / "gold_seed_v2.csv"
AUTO_JSONL = ROOT / "data" / "interim" / "labeled_auto.jsonl"
TO_LABEL = ROOT / "data" / "interim" / "to_label_v2.parquet"
OUT_DIR = ROOT / "data" / "processed"
DOCS_OUT = ROOT / "docs" / "dataset_card.md"

CONF_THRESHOLD = 0.5
TEST_FRAC = 0.15
VAL_FRAC = 0.15
LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

SEP = "=" * 60

print(SEP)
print("finalize_dataset.py — Building final dataset")
print(SEP)

if not GOLD_CSV.exists():
    print(f"ERROR: Not found: {GOLD_CSV}")
    sys.exit(1)

gold_raw = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
gold_raw["id"] = gold_raw["id"].astype(str)
gold_filled = gold_raw[
    gold_raw["label"].notna() & (gold_raw["label"].astype(str).str.strip() != "")
].copy()
gold_filled["label_human"] = gold_filled["label"].astype(str).str.strip().str.upper()
gold_filled = gold_filled[gold_filled["label_human"].isin(LABELS)]
gold_ids = set(gold_filled["id"].tolist())
print(f"\nGold seed: {len(gold_filled)}/150 articles with valid labels")
print(f"  dist: { {l: (gold_filled['label_human']==l).sum() for l in LABELS} }")

if not TO_LABEL.exists():
    print(f"ERROR: Not found: {TO_LABEL}")
    sys.exit(1)

tl = pd.read_parquet(TO_LABEL)
tl["id"] = tl["id"].astype(str)
tl["date"] = pd.to_datetime(tl["date"], errors="coerce")
print(f"\nto_label.parquet: {len(tl)} articles")

llm_df = pd.DataFrame()
if AUTO_JSONL.exists():
    recs = []
    with open(AUTO_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    llm_df = pd.DataFrame(recs)
    llm_df["id"] = llm_df["id"].astype(str)
    llm_df["label_llm"] = llm_df["label"].astype(str).str.strip().str.upper()
    llm_df["confidence"] = llm_df["confidence"].astype(float)
    llm_df = llm_df[llm_df["label_llm"].isin(LABELS)]
    n_before = len(llm_df)
    llm_df = llm_df[llm_df["confidence"] >= CONF_THRESHOLD]
    print(f"\nlabeled_auto.jsonl: {n_before} rows, {n_before - len(llm_df)} dropped "
          f"(conf < {CONF_THRESHOLD}), {len(llm_df)} remaining")
else:
    print("\nlabeled_auto.jsonl: not found — using gold seed only")

print("\nBuilding merged dataset...")

merged_parts = []

if len(llm_df) > 0:
    llm_merged = tl.merge(
        llm_df[["id", "label_llm", "confidence", "reason"]],
        on="id",
        how="inner",
    ).copy()
    llm_merged["label"] = llm_merged["label_llm"]
    llm_merged["label_source"] = "llm"
    llm_merged["confidence_val"] = llm_merged["confidence"]
    merged_parts.append(llm_merged.drop(columns=["label_llm", "confidence"], errors="ignore"))
    print(f"  LLM-labeled articles joined: {len(llm_merged)}")

gold_meta = tl[tl["id"].isin(gold_ids)].copy()
gold_labels = gold_filled[["id", "label_human", "note"]].copy()
gold_meta = gold_meta.merge(gold_labels, on="id", how="left")
gold_meta["label"] = gold_meta["label_human"]
gold_meta["label_source"] = "human"
gold_meta["confidence_val"] = 1.0
gold_meta["reason"] = gold_meta.get("note", "human annotation")
gold_meta = gold_meta.drop(columns=["label_human"], errors="ignore")
print(f"  Gold seed articles (human labels): {len(gold_meta)}")

merged_parts.append(gold_meta)

if not merged_parts:
    print("ERROR: No data to merge. Run prelabel_local.py first.")
    sys.exit(1)

df_all = pd.concat(merged_parts, ignore_index=True)

df_all["_sort"] = df_all["label_source"].map({"human": 0, "llm": 1})
df_all = df_all.sort_values("_sort").drop_duplicates(subset="id", keep="first")
df_all = df_all.drop(columns=["_sort"], errors="ignore")

print(f"\nMerged total (after dedup): {len(df_all)} articles")
print(f"  label_source: {df_all['label_source'].value_counts().to_dict()}")
print(f"  label dist:   { {l: (df_all['label']==l).sum() for l in LABELS} }")

print(f"\nTime-based split (test={TEST_FRAC:.0%}, val={VAL_FRAC:.0%}, train=rest)...")

df_sorted = df_all.sort_values("date", na_position="first").reset_index(drop=True)
n_all = len(df_sorted)

n_test = int(round(n_all * TEST_FRAC))
n_val = int(round(n_all * VAL_FRAC))

df_train = df_sorted.iloc[: n_all - n_test - n_val].copy()
df_val = df_sorted.iloc[n_all - n_test - n_val : n_all - n_test].copy()
df_test = df_sorted.iloc[n_all - n_test :].copy()

n_gold_test = df_test["id"].isin(gold_ids).sum()
print(f"  train : {len(df_train)} [{df_train['date'].min()} -> {df_train['date'].max()}]")
print(f"  val   : {len(df_val)} [{df_val['date'].min()} -> {df_val['date'].max()}]")
print(f"  test  : {len(df_test)} [{df_test['date'].min()} -> {df_test['date'].max()}]")
print(f"  gold in test: {n_gold_test} / {len(gold_ids)}")

if len(df_train) > 0 and len(df_val) > 0:
    assert df_train["date"].max() <= df_val["date"].min(), "OVERLAP train/val"
if len(df_val) > 0 and len(df_test) > 0:
    assert df_val["date"].max() <= df_test["date"].min(), "OVERLAP val/test"

BASE_COLS = ["id", "datetime", "date", "primary_ticker", "ticker_in_head",
             "title", "sapo", "text", "text_input",
             "label", "label_source", "confidence_val"]
extra_cols = [c for c in ["reason", "url"] if c in df_all.columns]
OUT_COLS = [c for c in BASE_COLS + extra_cols if c in df_all.columns]

OUT_DIR.mkdir(parents=True, exist_ok=True)

df_train_out = df_train[OUT_COLS].copy(); df_train_out["split"] = "train"
df_val_out = df_val[OUT_COLS].copy(); df_val_out["split"] = "val"
df_test_out = df_test[OUT_COLS].copy(); df_test_out["split"] = "test"
df_full_out = pd.concat([df_train_out, df_val_out, df_test_out], ignore_index=True)

df_train[OUT_COLS].to_parquet(OUT_DIR / "train.parquet", index=False)
df_val[OUT_COLS].to_parquet(OUT_DIR / "val.parquet", index=False)
df_test[OUT_COLS].to_parquet(OUT_DIR / "test.parquet", index=False)
df_full_out.to_parquet(OUT_DIR / "dataset_full.parquet", index=False)

print(f"\nFiles written:")
print(f"  {OUT_DIR / 'train.parquet'}")
print(f"  {OUT_DIR / 'val.parquet'}")
print(f"  {OUT_DIR / 'test.parquet'}")
print(f"  {OUT_DIR / 'dataset_full.parquet'}")


def split_stats(df: pd.DataFrame, name: str) -> dict:
    if len(df) == 0:
        return {"name": name, "n": 0}
    date_col = pd.to_datetime(df["date"], errors="coerce")
    return {
        "name": name,
        "n": len(df),
        "positive": (df["label"] == "POSITIVE").sum(),
        "neutral": (df["label"] == "NEUTRAL").sum(),
        "negative": (df["label"] == "NEGATIVE").sum(),
        "pos_pct": f"{(df['label']=='POSITIVE').mean()*100:.1f}%",
        "neu_pct": f"{(df['label']=='NEUTRAL').mean()*100:.1f}%",
        "neg_pct": f"{(df['label']=='NEGATIVE').mean()*100:.1f}%",
        "date_min": str(date_col.min())[:10],
        "date_max": str(date_col.max())[:10],
        "n_tickers": df["primary_ticker"].nunique(),
        "human": (df["label_source"] == "human").sum(),
        "llm": (df["label_source"] == "llm").sum(),
    }


splits_info = [
    split_stats(df_train, "train"),
    split_stats(df_val, "val"),
    split_stats(df_test, "test"),
    split_stats(df_full_out, "FULL"),
]

print()
print(SEP)
print("DATASET SUMMARY")
print(SEP)
header = f"{'Split':8s} {'N':>6} {'POS':>6} {'NEU':>6} {'NEG':>6}  {'Date Min':>10}  {'Date Max':>10}  {'Tickers':>7}  {'Human':>5}  {'LLM':>5}"
print(header)
print("-" * len(header))
for s in splits_info:
    if s["n"] == 0:
        continue
    print(
        f"{s['name']:8s} {s['n']:>6,} {s['positive']:>6} {s['neutral']:>6} {s['negative']:>6}"
        f"  {s['date_min']:>10}  {s['date_max']:>10}  {s['n_tickers']:>7}  {s['human']:>5}  {s['llm']:>5}"
    )

print()
print(SEP)
PASS = True

files_ok = all((OUT_DIR / f).exists() for f in ["train.parquet", "val.parquet", "test.parquet"])
if files_ok:
    print("PASS: 3 parquet files exist")
else:
    print("FAIL: missing parquet file")
    PASS = False

if len(df_train) >= 3500:
    print(f"PASS: train = {len(df_train):,} >= 3,500")
else:
    print(f"FAIL: train = {len(df_train):,} < 3,500")
    PASS = False

for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
    if len(df_split) == 0:
        print(f"FAIL: {split_name} is empty")
        PASS = False
        continue
    lbls = set(df_split["label"].unique())
    missing = set(LABELS) - lbls
    if not missing:
        print(f"PASS: {split_name} has all 3 labels")
    else:
        print(f"FAIL: {split_name} missing labels {missing}")
        PASS = False

if len(df_train) > 0 and len(df_val) > 0:
    t_max = pd.to_datetime(df_train["date"], errors="coerce").max()
    v_min = pd.to_datetime(df_val["date"], errors="coerce").min()
    if pd.isna(t_max) or pd.isna(v_min) or t_max <= v_min:
        print("PASS: no time overlap train/val")
    else:
        print(f"FAIL: overlap train/val ({t_max} > {v_min})")
        PASS = False

if len(df_val) > 0 and len(df_test) > 0:
    v_max = pd.to_datetime(df_val["date"], errors="coerce").max()
    t_min = pd.to_datetime(df_test["date"], errors="coerce").min()
    if pd.isna(v_max) or pd.isna(t_min) or v_max <= t_min:
        print("PASS: no time overlap val/test")
    else:
        print(f"FAIL: overlap val/test ({v_max} > {t_min})")
        PASS = False

gold_in_test = set(df_test["id"].astype(str)) & gold_ids
print(f"INFO: {len(gold_in_test)}/{len(gold_ids)} human-labeled articles in test "
      "(time-based split, no forced assignment)")

print(SEP)

DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)

with open(DOCS_OUT, "w", encoding="utf-8") as fh:
    def w(s=""):
        fh.write(s + "\n")

    w("# Dataset Card — VN Financial News Sentiment")
    w()
    w(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    w(f"**Task:** 3-class sentiment classification (POSITIVE / NEUTRAL / NEGATIVE)  ")
    w(f"**Language:** Vietnamese  ")
    w(f"**Domain:** Financial news (CafeF, VnExpress Business, etc.)  ")
    w()
    w("---")
    w()
    w("## Summary")
    w()
    w("| Attribute | Value |")
    w("|---|---|")
    w(f"| Total samples | {len(df_full_out):,} |")
    w(f"| Tickers | {df_full_out['primary_ticker'].nunique()} |")
    w(f"| Label sources | {(df_full_out['label_source']=='human').sum()} human + "
      f"{(df_full_out['label_source']=='llm').sum()} LLM |")
    w(f"| Date range | {str(pd.to_datetime(df_full_out['date'], errors='coerce').min())[:10]} → "
      f"{str(pd.to_datetime(df_full_out['date'], errors='coerce').max())[:10]} |")
    w(f"| Split strategy | Time-based (no leakage) |")
    w()
    w("---")
    w()
    w("## Splits")
    w()
    w("| Split | N | POSITIVE | NEUTRAL | NEGATIVE | Date Min | Date Max | Tickers | Human | LLM |")
    w("|-------|--:|--------:|---------:|---------:|----------|----------|--------:|------:|----:|")
    for s in splits_info:
        if s["n"] == 0:
            continue
        w(f"| {s['name']} | {s['n']:,} | {s['positive']} ({s['pos_pct']}) | "
          f"{s['neutral']} ({s['neu_pct']}) | {s['negative']} ({s['neg_pct']}) | "
          f"{s['date_min']} | {s['date_max']} | {s['n_tickers']} | {s['human']} | {s['llm']} |")
    w()
    w("---")
    w()
    w("## Label distribution (full dataset)")
    w()
    lbl_total = {l: (df_full_out["label"] == l).sum() for l in LABELS}
    w("| Label | Count | Rate |")
    w("|------|-------:|------:|")
    for lbl in LABELS:
        pct = lbl_total[lbl] / len(df_full_out) * 100
        w(f"| {lbl} | {lbl_total[lbl]:,} | {pct:.1f}% |")
    w()
    w("---")
    w()
    w("## Columns")
    w()
    w("| Column | Description |")
    w("|-----|-------|")
    w("| `id` | Article identifier |")
    w("| `date` | Publication date |")
    w("| `primary_ticker` | Primary stock ticker |")
    w("| `title` | Article headline |")
    w("| `sapo` | Lead paragraph |")
    w("| `text` | title + sapo (model input) |")
    w("| `label` | Sentiment label: POSITIVE / NEUTRAL / NEGATIVE |")
    w("| `label_source` | Label origin: `human` or `llm` |")
    w("| `confidence_val` | Label confidence (1.0 for human) |")
    if "reason" in df_full_out.columns:
        w("| `reason` | Labeling rationale (from LLM) |")
    if "url" in df_full_out.columns:
        w("| `url` | Original article URL |")
    w("| `split` | train / val / test |")
    w()
    w("---")
    w()
    w("## Dataset construction pipeline")
    w()
    w("1. **Crawl:** scrape Vietnamese financial news → `data/raw/articles.jsonl`")
    w("2. **Clean:** dedup URLs and headlines, filter dates, assign tickers → `data/interim/articles_clean.parquet`")
    w("3. **Sample:** stratified by `primary_ticker`, max 60 per ticker → `data/interim/to_label.parquet`")
    w("4. **Auto-label:** batch inference, 3 retries → `data/interim/labeled_auto.jsonl`")
    w("5. **Gold seed:** 150 articles labeled by human following `docs/labeling_guide.md` → `data/interim/gold_seed.csv`")
    w("6. **Merge & split:** human labels take priority; time-based split → `data/processed/`")
    w()
    w("---")
    w()
    w("> **Note:** Random split is not used to avoid future information leaking into the train set.")
    w("> The gold seed (150 human-labeled articles) is in the test set for evaluation on highest-quality labels.")

print(f"\nDataset card written: {DOCS_OUT}")

if not PASS:
    sys.exit(1)