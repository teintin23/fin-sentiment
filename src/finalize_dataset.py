"""
finalize_dataset.py
-------------------
Tạo dataset cuối cùng để train mô hình sentiment.

Inputs:
  data/interim/to_label.parquet      — 6000 bài cần nhãn
  data/interim/labeled_auto.jsonl     — nhãn máy (tuỳ chọn, nếu chưa có thì bỏ qua)
  data/interim/gold_seed.csv         — 150 nhãn người (bắt buộc)

Outputs:
  data/processed/train.parquet
  data/processed/val.parquet
  data/processed/test.parquet
  data/processed/dataset_full.parquet
  docs/dataset_card.md

Split logic (time-based, no leakage):
  - Gold seed → test  (ưu tiên tuyệt đối, bất kể ngày)
  - Còn lại sắp theo ngày: test=15% mới nhất, val=15% tiếp, train=70% cũ nhất

PASS khi:
  - 3 file parquet tồn tại
  - train >= 3500 dòng
  - Mỗi split có đủ 3 nhãn
  - Khoảng thời gian không chồng lấn giữa các split

Usage:
    python src/finalize_dataset.py
"""

from __future__ import annotations

import io
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
GOLD_CSV = ROOT / "data" / "interim" / "gold_seed_v2.csv"
AUTO_JSONL = ROOT / "data" / "interim" / "labeled_auto.jsonl"
TO_LABEL  = ROOT / "data" / "interim" / "to_label_v2.parquet"
OUT_DIR   = ROOT / "data" / "processed"
DOCS_OUT  = ROOT / "docs" / "dataset_card.md"

CONF_THRESHOLD = 0.5
TEST_FRAC = 0.15
VAL_FRAC  = 0.15
LABELS    = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

SEP = "=" * 60

# ---------------------------------------------------------------------------
# 0. Load gold seed (required)
# ---------------------------------------------------------------------------
print(SEP)
print("finalize_dataset.py — Tao dataset cuoi cung")
print(SEP)

if not GOLD_CSV.exists():
    print(f"ERROR: Khong tim thay {GOLD_CSV}")
    sys.exit(1)

gold_raw = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
gold_raw["id"] = gold_raw["id"].astype(str)
gold_filled = gold_raw[
    gold_raw["label"].notna() & (gold_raw["label"].astype(str).str.strip() != "")
].copy()
gold_filled["label_human"] = gold_filled["label"].astype(str).str.strip().str.upper()
gold_filled = gold_filled[gold_filled["label_human"].isin(LABELS)]
gold_ids = set(gold_filled["id"].tolist())
print(f"\nGold seed: {len(gold_filled)}/150 bai co nhan hop le")
print(f"  dist: { {l: (gold_filled['label_human']==l).sum() for l in LABELS} }")

# ---------------------------------------------------------------------------
# 1. Load to_label (article metadata)
# ---------------------------------------------------------------------------
if not TO_LABEL.exists():
    print(f"ERROR: Khong tim thay {TO_LABEL}")
    sys.exit(1)

tl = pd.read_parquet(TO_LABEL)
tl["id"] = tl["id"].astype(str)
tl["date"] = pd.to_datetime(tl["date"], errors="coerce")
print(f"\nto_label.parquet: {len(tl)} bai")

# ---------------------------------------------------------------------------
# 2. Load LLM labels (optional)
# ---------------------------------------------------------------------------
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
    # Filter valid labels
    llm_df = llm_df[llm_df["label_llm"].isin(LABELS)]
    n_before = len(llm_df)
    # Step 3: Drop low-confidence
    llm_df = llm_df[llm_df["confidence"] >= CONF_THRESHOLD]
    print(f"\nlabeled_auto.jsonl: {n_before} rows, {n_before - len(llm_df)} dropped (conf < {CONF_THRESHOLD}), {len(llm_df)} remaining")
else:
    print("\nlabeled_auto.jsonl: khong tim thay — chi dung gold seed")

# ---------------------------------------------------------------------------
# 3. Build merged dataset
# ---------------------------------------------------------------------------
print("\nBuilding merged dataset...")

# Start with to_label as base (for metadata: date, title, sapo, text, etc.)
merged_parts = []

if len(llm_df) > 0:
    # Join LLM labels onto article metadata
    llm_merged = tl.merge(
        llm_df[["id", "label_llm", "confidence", "reason"]],
        on="id",
        how="inner",
    ).copy()
    llm_merged["label"]        = llm_merged["label_llm"]
    llm_merged["label_source"] = "llm"
    llm_merged["confidence_val"] = llm_merged["confidence"]
    merged_parts.append(llm_merged.drop(columns=["label_llm", "confidence"], errors="ignore"))
    print(f"  LLM-labeled articles joined: {len(llm_merged)}")

# Step 2: Override gold seed with human labels
# Join gold human labels onto article metadata
gold_meta = tl[tl["id"].isin(gold_ids)].copy()
gold_labels = gold_filled[["id", "label_human", "note"]].copy()
gold_meta = gold_meta.merge(gold_labels, on="id", how="left")
gold_meta["label"]        = gold_meta["label_human"]
gold_meta["label_source"] = "human"
gold_meta["confidence_val"] = 1.0
gold_meta["reason"]       = gold_meta.get("note", "human annotation")
gold_meta = gold_meta.drop(columns=["label_human"], errors="ignore")
print(f"  Gold seed articles (human labels): {len(gold_meta)}")

merged_parts.append(gold_meta)

if not merged_parts:
    print("ERROR: Khong co du lieu nao de merge. Chay prelabel_local.py truoc.")
    sys.exit(1)

# Concatenate and deduplicate — gold (human) takes priority over LLM
df_all = pd.concat(merged_parts, ignore_index=True)

# When both LLM and human labels exist for same id, keep human
df_all["_sort"] = df_all["label_source"].map({"human": 0, "llm": 1})
df_all = df_all.sort_values("_sort").drop_duplicates(subset="id", keep="first")
df_all = df_all.drop(columns=["_sort"], errors="ignore")

print(f"\nMerged total (after dedup): {len(df_all)} bai")
print(f"  label_source: {df_all['label_source'].value_counts().to_dict()}")
print(f"  label dist:   { {l: (df_all['label']==l).sum() for l in LABELS} }")

# ---------------------------------------------------------------------------
# 4. Time-based split
# ---------------------------------------------------------------------------
print(f"\nTime-based split (test={TEST_FRAC:.0%}, val={VAL_FRAC:.0%}, train=rest)...")

# PURE time-based split. Gold KHONG bi ep vao test — ep gold (rai deu theo
# thoi gian) vao test se pha vo ranh gioi thoi gian va gay ro ri train/test.
# Gold van duoc danh dau qua cot label_source == "human" de bao cao rieng.
df_sorted = df_all.sort_values("date", na_position="first").reset_index(drop=True)
n_all = len(df_sorted)

n_test = int(round(n_all * TEST_FRAC))
n_val  = int(round(n_all * VAL_FRAC))

df_train = df_sorted.iloc[: n_all - n_test - n_val].copy()
df_val   = df_sorted.iloc[n_all - n_test - n_val : n_all - n_test].copy()
df_test  = df_sorted.iloc[n_all - n_test :].copy()

n_gold_test = df_test["id"].isin(gold_ids).sum()
print(f"  train : {len(df_train)} bai  [{df_train['date'].min()} -> {df_train['date'].max()}]")
print(f"  val   : {len(df_val)} bai  [{df_val['date'].min()} -> {df_val['date'].max()}]")
print(f"  test  : {len(df_test)} bai  [{df_test['date'].min()} -> {df_test['date'].max()}]")
print(f"  gold roi vao test: {n_gold_test} / {len(gold_ids)}")

if len(df_train) > 0 and len(df_val) > 0:
    assert df_train["date"].max() <= df_val["date"].min(), "OVERLAP train/val"
if len(df_val) > 0 and len(df_test) > 0:
    assert df_val["date"].max() <= df_test["date"].min(), "OVERLAP val/test"

# ---------------------------------------------------------------------------
# 5. Write parquet files
# ---------------------------------------------------------------------------
# Select output columns
BASE_COLS = ["id", "datetime", "date", "primary_ticker", "ticker_in_head",
             "title", "sapo", "text", "text_input",
             "label", "label_source", "confidence_val"]
extra_cols = [c for c in ["reason", "url"] if c in df_all.columns]
OUT_COLS = [c for c in BASE_COLS + extra_cols if c in df_all.columns]

OUT_DIR.mkdir(parents=True, exist_ok=True)

df_full  = pd.concat([df_train, df_val, df_test], ignore_index=True)

# Add split column to full dataset
df_train_out = df_train[OUT_COLS].copy(); df_train_out["split"] = "train"
df_val_out   = df_val[OUT_COLS].copy();   df_val_out["split"]   = "val"
df_test_out  = df_test[OUT_COLS].copy();  df_test_out["split"]  = "test"
df_full_out  = pd.concat([df_train_out, df_val_out, df_test_out], ignore_index=True)

df_train[OUT_COLS].to_parquet(OUT_DIR / "train.parquet", index=False)
df_val[OUT_COLS].to_parquet(OUT_DIR / "val.parquet",     index=False)
df_test[OUT_COLS].to_parquet(OUT_DIR / "test.parquet",   index=False)
df_full_out.to_parquet(OUT_DIR / "dataset_full.parquet", index=False)

print(f"\nFiles written:")
print(f"  {OUT_DIR / 'train.parquet'}")
print(f"  {OUT_DIR / 'val.parquet'}")
print(f"  {OUT_DIR / 'test.parquet'}")
print(f"  {OUT_DIR / 'dataset_full.parquet'}")

# ---------------------------------------------------------------------------
# 6. Print summary table
# ---------------------------------------------------------------------------
def split_stats(df: pd.DataFrame, name: str) -> dict:
    if len(df) == 0:
        return {"name": name, "n": 0}
    date_col = pd.to_datetime(df["date"], errors="coerce")
    return {
        "name":     name,
        "n":        len(df),
        "positive": (df["label"] == "POSITIVE").sum(),
        "neutral":  (df["label"] == "NEUTRAL").sum(),
        "negative": (df["label"] == "NEGATIVE").sum(),
        "pos_pct":  f"{(df['label']=='POSITIVE').mean()*100:.1f}%",
        "neu_pct":  f"{(df['label']=='NEUTRAL').mean()*100:.1f}%",
        "neg_pct":  f"{(df['label']=='NEGATIVE').mean()*100:.1f}%",
        "date_min": str(date_col.min())[:10],
        "date_max": str(date_col.max())[:10],
        "n_tickers": df["primary_ticker"].nunique(),
        "human":    (df["label_source"] == "human").sum(),
        "llm":      (df["label_source"] == "llm").sum(),
    }

splits_info = [
    split_stats(df_train, "train"),
    split_stats(df_val,   "val"),
    split_stats(df_test,  "test"),
    split_stats(df_full_out, "FULL"),
]

print()
print(SEP)
print("TONG KET DATASET")
print(SEP)
header = f"{'Split':8s} {'N':>6} {'POS':>6} {'NEU':>6} {'NEG':>6}  {'Ngay min':>10}  {'Ngay max':>10}  {'Tickers':>7}  {'Human':>5}  {'LLM':>5}"
print(header)
print("-" * len(header))
for s in splits_info:
    if s["n"] == 0:
        continue
    print(
        f"{s['name']:8s} {s['n']:>6,} {s['positive']:>6} {s['neutral']:>6} {s['negative']:>6}"
        f"  {s['date_min']:>10}  {s['date_max']:>10}  {s['n_tickers']:>7}  {s['human']:>5}  {s['llm']:>5}"
    )

# ---------------------------------------------------------------------------
# 7. PASS / FAIL
# ---------------------------------------------------------------------------
print()
print(SEP)
PASS = True

files_ok = all((OUT_DIR / f).exists() for f in ["train.parquet", "val.parquet", "test.parquet"])
if files_ok:
    print("PASS: 3 file parquet ton tai")
else:
    print("FAIL: thieu file parquet")
    PASS = False

if len(df_train) >= 3500:
    print(f"PASS: train = {len(df_train):,} >= 3,500")
else:
    print(f"FAIL: train = {len(df_train):,} < 3,500")
    PASS = False

for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
    if len(df_split) == 0:
        print(f"FAIL: {split_name} rong")
        PASS = False
        continue
    lbls = set(df_split["label"].unique())
    missing = set(LABELS) - lbls
    if not missing:
        print(f"PASS: {split_name} co du 3 nhan")
    else:
        print(f"FAIL: {split_name} thieu nhan {missing}")
        PASS = False

# Check no time overlap on non-gold articles
if len(df_train) > 0 and len(df_val) > 0:
    t_max = pd.to_datetime(df_train["date"], errors="coerce").max()
    v_min = pd.to_datetime(df_val["date"], errors="coerce").min()
    if pd.isna(t_max) or pd.isna(v_min) or t_max <= v_min:
        print("PASS: khong chong lan thoi gian train/val")
    else:
        print(f"FAIL: chong lan train/val ({t_max} > {v_min})")
        PASS = False

if len(df_val) > 0 and len(df_test) > 0:
    v_max = pd.to_datetime(df_val["date"], errors="coerce").max()
    t_min = pd.to_datetime(df_test["date"], errors="coerce").min()
    if pd.isna(v_max) or pd.isna(t_min) or v_max <= t_min:
        print("PASS: khong chong lan thoi gian val/test")
    else:
        print(f"FAIL: chong lan val/test ({v_max} > {t_min})")
        PASS = False

gold_in_test = set(df_test["id"].astype(str)) & gold_ids
print(f"INFO: {len(gold_in_test)}/{len(gold_ids)} nhan nguoi roi vao test "
      f"(split thuan thoi gian, khong ep gold vao test)")


print(SEP)

# ---------------------------------------------------------------------------
# 8. Write dataset_card.md
# ---------------------------------------------------------------------------
DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)

with open(DOCS_OUT, "w", encoding="utf-8") as fh:
    def w(s=""):
        fh.write(s + "\n")

    w("# Dataset Card — VN Financial News Sentiment")
    w()
    w(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    w(f"**Task:** 3-class sentiment classification (POSITIVE / NEUTRAL / NEGATIVE)  ")
    w(f"**Language:** Vietnamese  ")
    w(f"**Domain:** Financial news (CafeF, VnExpress Business, v.v.)  ")
    w()
    w("---")
    w()
    w("## Tổng quan")
    w()
    w("| Thuộc tính | Chi tiết |")
    w("|---|---|")
    w(f"| Tổng số mẫu | {len(df_full_out):,} |")
    w(f"| Số mã cổ phiếu | {df_full_out['primary_ticker'].nunique()} |")
    w(f"| Nguồn nhãn | {(df_full_out['label_source']=='human').sum()} nhân + {(df_full_out['label_source']=='llm').sum()} LLM |")
    w(f"| Khoảng thời gian | {str(pd.to_datetime(df_full_out['date'], errors='coerce').min())[:10]} → {str(pd.to_datetime(df_full_out['date'], errors='coerce').max())[:10]} |")
    w(f"| Split strategy | Time-based (no leakage), gold seed → test |")
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
        w(f"| {s['name']} | {s['n']:,} | {s['positive']} ({s['pos_pct']}) | {s['neutral']} ({s['neu_pct']}) | {s['negative']} ({s['neg_pct']}) | {s['date_min']} | {s['date_max']} | {s['n_tickers']} | {s['human']} | {s['llm']} |")
    w()
    w("---")
    w()
    w("## Phân bố nhãn toàn bộ dataset")
    w()
    lbl_total = {l: (df_full_out["label"]==l).sum() for l in LABELS}
    w("| Nhãn | Số mẫu | Tỷ lệ |")
    w("|------|-------:|------:|")
    for lbl in LABELS:
        pct = lbl_total[lbl] / len(df_full_out) * 100
        w(f"| {lbl} | {lbl_total[lbl]:,} | {pct:.1f}% |")
    w()
    w("---")
    w()
    w("## Columns")
    w()
    w("| Cột | Mô tả |")
    w("|-----|-------|")
    w("| `id` | ID bài báo gốc |")
    w("| `date` | Ngày đăng |")
    w("| `primary_ticker` | Mã cổ phiếu chính |")
    w("| `title` | Tiêu đề bài báo |")
    w("| `sapo` | Đoạn dẫn (sapo) |")
    w("| `text` | title + sapo (input cho mô hình) |")
    w("| `label` | Nhãn sentiment: POSITIVE / NEUTRAL / NEGATIVE |")
    w("| `label_source` | Nguồn nhãn: `human` hoặc `llm` |")
    w("| `confidence_val` | Độ tin cậy của nhãn (1.0 nếu human) |")
    if "reason" in df_full_out.columns:
        w("| `reason` | Lý do gán nhãn (từ LLM) |")
    if "url" in df_full_out.columns:
        w("| `url` | URL bài báo gốc |")
    w("| `split` | train / val / test |")
    w()
    w("---")
    w()
    w("## Quy trình xây dựng dataset")
    w()
    w("1. **Thu thập:** Crawl từ báo tài chính tiếng Việt → `data/raw/articles.jsonl`")
    w("2. **Làm sạch:** Dedupe URL + tiêu đề chuẩn hóa, lọc ngày, tạo cột tickers → `data/interim/articles_clean.parquet`")
    w("3. **Sampling:** Phân tầng theo `primary_ticker`, tối đa 60 bài/mã → `data/interim/to_label.parquet`")
    w("4. **Gán nhãn máy:** Batch 20 bài/request, retry 3 lần → `data/interim/labeled_auto.jsonl`")
    w("5. **Gold seed:** 150 bài được gán nhãn tay theo `docs/labeling_guide.md` → `data/interim/gold_seed.csv`")
    w("6. **Merge & split:** Nhãn người ưu tiên LLM; split time-based; gold seed → test → `data/processed/`")
    w()
    w("---")
    w()
    w("> **Lưu ý:** Không dùng random split để tránh rò rỉ thông tin tương lai vào train set.")
    w("> Gold seed (150 bài nhãn người) nằm hoàn toàn trong test set để đánh giá mô hình trên nhãn chất lượng cao nhất.")

print(f"\nDataset card written: {DOCS_OUT}")

if not PASS:
    sys.exit(1)