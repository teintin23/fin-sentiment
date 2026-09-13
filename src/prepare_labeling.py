from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "interim" / "articles_clean.parquet"
OUT_LABEL = ROOT / "data" / "interim" / "to_label.parquet"
OUT_GOLD_PQ = ROOT / "data" / "interim" / "gold_seed.parquet"
OUT_GOLD_CSV = ROOT / "data" / "interim" / "gold_seed.csv"

RANDOM_STATE = 42
TOTAL_TARGET = 6000
MAX_PER_TICKER = 60
GOLD_SIZE = 150

LABEL_COLS = ["id", "date", "primary_ticker", "title", "sapo", "text", "url"]


def _has_widget(v) -> bool:
    try:
        return len(v) > 0
    except Exception:
        return False


print("=" * 60)
print("Loading articles_clean.parquet ...")
df = pd.read_parquet(IN_FILE)
print(f"  Total rows: {len(df):,}")

print("\nStep 1 - Filter is_macro == False ...")
df = df[df["is_macro"] == False].copy()
print(f"  -> {len(df):,} rows with ticker")

print("Step 2 - Filter primary_ticker not null ...")
df = df[df["primary_ticker"].notna()].copy()
print(f"  -> {len(df):,} rows with primary_ticker set")

print(
    f"\nStep 3 - Stratified sampling "
    f"(target={TOTAL_TARGET:,}, max_per_ticker={MAX_PER_TICKER}) ..."
)

df["_has_widget"] = df["tickers_widget"].apply(_has_widget)

sampled_parts: list[pd.DataFrame] = []

for ticker, group in df.groupby("primary_ticker", sort=False):
    n_take = min(len(group), MAX_PER_TICKER)

    widget_rows = group[group["_has_widget"] == True]
    other_rows = group[group["_has_widget"] == False]

    if len(widget_rows) >= n_take:
        chosen = widget_rows.sample(n=n_take, random_state=RANDOM_STATE)
    elif len(widget_rows) > 0:
        remainder = n_take - len(widget_rows)
        fill = other_rows.sample(
            n=min(remainder, len(other_rows)),
            random_state=RANDOM_STATE,
        )
        chosen = pd.concat([widget_rows, fill])
    else:
        chosen = group.sample(n=n_take, random_state=RANDOM_STATE)

    sampled_parts.append(chosen)

sampled = pd.concat(sampled_parts, ignore_index=True)
print(f"  -> {len(sampled):,} rows after per-ticker cap")

if len(sampled) > TOTAL_TARGET:
    sampled = sampled.sample(n=TOTAL_TARGET, random_state=RANDOM_STATE)
    print(f"  -> trimmed to {len(sampled):,}")

sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

keep = [c for c in LABEL_COLS if c in sampled.columns]
to_label = sampled[keep].copy()

OUT_LABEL.parent.mkdir(parents=True, exist_ok=True)
to_label.to_parquet(OUT_LABEL, index=False, engine="pyarrow")
print(f"\n  Written: {OUT_LABEL}  ({len(to_label):,} rows)")

print("\n" + "=" * 60)
print("SUMMARY - to_label.parquet")
print("=" * 60)

print(f"\nArticles       : {len(to_label):,}")
print(f"Unique tickers : {to_label['primary_ticker'].nunique():,}")

print("\nTop 15 tickers:")
top15 = to_label["primary_ticker"].value_counts().head(15)
for ticker, cnt in top15.items():
    bar = "#" * int(cnt / 2)
    print(f"  {ticker:8s} {cnt:4d}  {bar}")

print("\nDistribution by year:")
_dates = pd.to_datetime(to_label["date"], errors="coerce")
year_dist = _dates.dt.year.value_counts().sort_index()
for yr, cnt in year_dist.items():
    bar = "#" * int(cnt / 15)
    print(f"  {yr:.0f}  {cnt:5,}  {bar}")

print("\n" + "=" * 60)
print(f"Extracting gold seed ({GOLD_SIZE} articles) ...")

gold = to_label.sample(n=GOLD_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

gold.insert(len(gold.columns), "label", "")
gold.insert(len(gold.columns), "note", "")

gold_pq = gold.drop(columns=["label", "note"])
gold_pq.to_parquet(OUT_GOLD_PQ, index=False, engine="pyarrow")
print(f"  Written: {OUT_GOLD_PQ}  ({len(gold_pq):,} rows)")

gold.to_csv(OUT_GOLD_CSV, index=False, encoding="utf-8-sig")
print(f"  Written: {OUT_GOLD_CSV}  ({len(gold):,} rows, utf-8-sig)")

print("\n" + "=" * 60)
PASS = True

to_label_chk = pd.read_parquet(OUT_LABEL)
gold_chk = pd.read_csv(OUT_GOLD_CSV, encoding="utf-8-sig")

if len(to_label_chk) == TOTAL_TARGET:
    print(f"PASS: to_label.parquet has {len(to_label_chk):,} rows")
else:
    print(
        f"FAIL: to_label.parquet has {len(to_label_chk):,} rows "
        f"(expected {TOTAL_TARGET:,})"
    )
    PASS = False

if len(gold_chk) == GOLD_SIZE:
    print(f"PASS: gold_seed.csv has {len(gold_chk):,} rows")
else:
    print(f"FAIL: gold_seed.csv has {len(gold_chk):,} rows (expected {GOLD_SIZE})")
    PASS = False

if "label" in gold_chk.columns and "note" in gold_chk.columns:
    print("PASS: gold_seed.csv has 'label' and 'note' columns")
else:
    print("FAIL: gold_seed.csv missing label/note columns")
    PASS = False

title_sample = gold_chk["title"].iloc[0]
if any(ord(c) > 127 for c in str(title_sample)):
    print("PASS: UTF-8 Vietnamese text OK")
else:
    print("WARN: no Vietnamese characters found in title sample")

print("=" * 60)

if not PASS:
    sys.exit(1)
