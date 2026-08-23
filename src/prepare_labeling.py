"""
prepare_labeling.py
-------------------
Chuẩn bị tập cần gán nhãn từ data/interim/articles_clean.parquet.

Output:
  data/interim/to_label.parquet   -- 6000 bài phân tầng theo primary_ticker
  data/interim/gold_seed.parquet  -- 150 bài mẫu để gán nhãn tay
  data/interim/gold_seed.csv      -- bản CSV cho Excel (utf-8-sig)

Usage:
    python src/prepare_labeling.py
"""

from __future__ import annotations

import io
import sys

# Force UTF-8 output so Vietnamese prints correctly on any console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
IN_FILE      = ROOT / "data" / "interim" / "articles_clean.parquet"
OUT_LABEL    = ROOT / "data" / "interim" / "to_label.parquet"
OUT_GOLD_PQ  = ROOT / "data" / "interim" / "gold_seed.parquet"
OUT_GOLD_CSV = ROOT / "data" / "interim" / "gold_seed.csv"

RANDOM_STATE   = 42
TOTAL_TARGET   = 6000
MAX_PER_TICKER = 60
GOLD_SIZE      = 150

LABEL_COLS = ["id", "date", "primary_ticker", "title", "sapo", "text", "url"]


# -- Helper --------------------------------------------------------------------
def _has_widget(v) -> bool:
    """Return True if the tickers_widget array/list is non-empty."""
    try:
        return len(v) > 0
    except Exception:
        return False


# -- Load ----------------------------------------------------------------------
print("=" * 60)
print("Loading articles_clean.parquet ...")
df = pd.read_parquet(IN_FILE)
print(f"  Total rows in parquet : {len(df):,}")

# -- Step 1: Filter is_macro == False ------------------------------------------
print("\nStep 1 - Filter is_macro == False ...")
df = df[df["is_macro"] == False].copy()
print(f"  -> {len(df):,} rows with a ticker")

# -- Step 2: Filter primary_ticker not null ------------------------------------
print("Step 2 - Filter primary_ticker not null ...")
df = df[df["primary_ticker"].notna()].copy()
print(f"  -> {len(df):,} rows with primary_ticker set")

# -- Step 3: Stratified sampling -----------------------------------------------
print(
    f"\nStep 3 - Stratified sampling "
    f"(target={TOTAL_TARGET:,}, max_per_ticker={MAX_PER_TICKER}) ..."
)

# Mark rows that have a tickers_widget entry (highest-confidence source)
df["_has_widget"] = df["tickers_widget"].apply(_has_widget)

sampled_parts: list[pd.DataFrame] = []

for ticker, group in df.groupby("primary_ticker", sort=False):
    n_take = min(len(group), MAX_PER_TICKER)

    widget_rows = group[group["_has_widget"] == True]
    other_rows  = group[group["_has_widget"] == False]

    if len(widget_rows) >= n_take:
        # Enough high-confidence rows
        chosen = widget_rows.sample(n=n_take, random_state=RANDOM_STATE)
    elif len(widget_rows) > 0:
        # Take all widget rows, fill remainder from others
        remainder = n_take - len(widget_rows)
        fill = other_rows.sample(
            n=min(remainder, len(other_rows)),
            random_state=RANDOM_STATE,
        )
        chosen = pd.concat([widget_rows, fill])
    else:
        # No widget rows — pure random
        chosen = group.sample(n=n_take, random_state=RANDOM_STATE)

    sampled_parts.append(chosen)

sampled = pd.concat(sampled_parts, ignore_index=True)
print(f"  -> {len(sampled):,} rows after per-ticker cap (before final trim)")

# Trim to exactly TOTAL_TARGET if we got more
if len(sampled) > TOTAL_TARGET:
    sampled = sampled.sample(n=TOTAL_TARGET, random_state=RANDOM_STATE)
    print(f"  -> trimmed to {len(sampled):,}")

# Final shuffle
sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# -- Select output columns -----------------------------------------------------
keep = [c for c in LABEL_COLS if c in sampled.columns]
to_label = sampled[keep].copy()

# -- Write to_label.parquet ----------------------------------------------------
OUT_LABEL.parent.mkdir(parents=True, exist_ok=True)
to_label.to_parquet(OUT_LABEL, index=False, engine="pyarrow")
print(f"\n  Written: {OUT_LABEL}  ({len(to_label):,} rows)")

# -- Step 4: Summary statistics -----------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY - to_label.parquet")
print("=" * 60)

print(f"\nSo bai              : {len(to_label):,}")
print(f"So ma khac nhau     : {to_label['primary_ticker'].nunique():,}")

print("\nTop 15 ma:")
top15 = to_label["primary_ticker"].value_counts().head(15)
for ticker, cnt in top15.items():
    bar = "#" * int(cnt / 2)
    print(f"  {ticker:8s} {cnt:4d}  {bar}")

print("\nPhan bo theo nam:")
_dates = pd.to_datetime(to_label["date"], errors="coerce")
year_dist = _dates.dt.year.value_counts().sort_index()
for yr, cnt in year_dist.items():
    bar = "#" * int(cnt / 15)
    print(f"  {yr:.0f}  {cnt:5,}  {bar}")

# -- Step 5: Gold seed --------------------------------------------------------
print("\n" + "=" * 60)
print(f"Tach gold_seed ({GOLD_SIZE} bai) ...")

gold = to_label.sample(n=GOLD_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

# Add empty annotation columns
gold.insert(len(gold.columns), "label", "")
gold.insert(len(gold.columns), "note", "")

# Parquet — no annotation cols
gold_pq = gold.drop(columns=["label", "note"])
gold_pq.to_parquet(OUT_GOLD_PQ, index=False, engine="pyarrow")
print(f"  Written: {OUT_GOLD_PQ}  ({len(gold_pq):,} rows)")

# CSV with utf-8-sig (BOM so Excel opens correctly)
gold.to_csv(OUT_GOLD_CSV, index=False, encoding="utf-8-sig")
print(f"  Written: {OUT_GOLD_CSV}  ({len(gold):,} rows, utf-8-sig)")

# -- PASS / FAIL ---------------------------------------------------------------
print("\n" + "=" * 60)
PASS = True

# Verify by re-reading
to_label_chk = pd.read_parquet(OUT_LABEL)
gold_chk = pd.read_csv(OUT_GOLD_CSV, encoding="utf-8-sig")

if len(to_label_chk) == TOTAL_TARGET:
    print(f"PASS: to_label.parquet co {len(to_label_chk):,} dong")
else:
    print(
        f"FAIL: to_label.parquet co {len(to_label_chk):,} dong "
        f"(yeu cau {TOTAL_TARGET:,})"
    )
    PASS = False

if len(gold_chk) == GOLD_SIZE:
    print(f"PASS: gold_seed.csv co {len(gold_chk):,} dong")
else:
    print(f"FAIL: gold_seed.csv co {len(gold_chk):,} dong (yeu cau {GOLD_SIZE})")
    PASS = False

if "label" in gold_chk.columns and "note" in gold_chk.columns:
    print("PASS: gold_seed.csv co cot 'label' va 'note'")
else:
    print("FAIL: thieu cot label/note trong gold_seed.csv")
    PASS = False

# Vietnamese font check
title_sample = gold_chk["title"].iloc[0]
if any(ord(c) > 127 for c in str(title_sample)):
    print("PASS: UTF-8 tieng Viet doc lai OK")
else:
    print("WARN: khong tim thay ky tu tieng Viet trong title mau")

print("=" * 60)

if not PASS:
    sys.exit(1)
