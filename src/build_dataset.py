from __future__ import annotations

import json
import re
import string
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "articles.jsonl"
OUT_FILE = ROOT / "data" / "interim" / "articles_clean.parquet"
DOCS_FILE = ROOT / "docs" / "data_quality.md"

TODAY = date.today()
MIN_DATE = date(2015, 1, 1)

_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + r"]")
_WS_RE = re.compile(r"\s+")


def normalise_title(title: str) -> str:
    t = title.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def first_elem(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return lst[0]
    return None


print("=" * 60)
print("Step 1 - Loading articles.jsonl ...")
records = []
with open(RAW_FILE, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

df = pd.DataFrame(records)
step_counts = []
n = len(df)
step_counts.append(("1 - Total loaded", n))
print(f"  -> {n:,} rows")

print("Step 2 - Drop rows where title is empty or len(title) < 15 ...")
df["title"] = df["title"].fillna("").astype(str).str.strip()
df = df[df["title"].str.len() >= 15].copy()
n = len(df)
step_counts.append(("2 - After title filter", n))
print(f"  -> {n:,} rows")

print("Step 3 - Deduplicate by url ...")
df["url"] = df["url"].fillna("").astype(str).str.strip()
df = df.drop_duplicates(subset=["url"], keep="first").copy()
n = len(df)
step_counts.append(("3 - After url dedup", n))
print(f"  -> {n:,} rows")

print("Step 4 - Deduplicate by normalised title (keep earliest date) ...")
df["title_norm"] = df["title"].apply(normalise_title)
df["_date_sort"] = df["date"].fillna("9999-99-99").astype(str)
df = df.sort_values("_date_sort", ascending=True)
df = df.drop_duplicates(subset=["title_norm"], keep="first").copy()
df = df.drop(columns=["_date_sort"])
n = len(df)
step_counts.append(("4 - After title dedup", n))
print(f"  -> {n:,} rows")

print("Step 5 - Parse date + time -> datetime ...")
df["date"] = df["date"].fillna("").astype(str).str.strip()
df["time"] = df["time"].fillna("").astype(str).str.strip()
df["_dt_str"] = df.apply(
    lambda r: (r["date"] + " " + r["time"]).strip() if r["date"] else "",
    axis=1,
)
df["datetime"] = pd.to_datetime(df["_dt_str"], format="mixed", errors="coerce")
df = df.drop(columns=["_dt_str"])
print(f"  -> {len(df):,} rows (datetime parse done)")

print("Step 6 - Drop rows with null/future/pre-2015 dates ...")
dt_date = df["datetime"].dt.date
mask = (
    df["datetime"].notna()
    & (dt_date >= MIN_DATE)
    & (dt_date <= TODAY)
)
df = df[mask].copy()
n = len(df)
step_counts.append(("5 - After null date filter", n))
step_counts.append(("6 - After date range filter", n))
print(f"  -> {n:,} rows")

print("Step 7 - Create text = title + sapo ...")
df["sapo"] = df["sapo"].fillna("").astype(str).str.strip()
df["text"] = (df["title"] + " " + df["sapo"]).apply(lambda s: _WS_RE.sub(" ", s).strip())

print("Step 8 - Create n_tickers ...")

def safe_len(v):
    return len(v) if isinstance(v, list) else 0

for col in ("tickers_any", "tickers_widget", "tickers_explicit", "tickers_by_name"):
    if col not in df.columns:
        df[col] = [[] for _ in range(len(df))]
    else:
        df[col] = df[col].apply(lambda v: v if isinstance(v, list) else [])

df["n_tickers"] = df["tickers_any"].apply(safe_len)

print("Step 9 - Create is_macro ...")
df["is_macro"] = df["n_tickers"] == 0

print("Step 10 - Create primary_ticker ...")

def primary_ticker(row):
    for col in ("tickers_widget", "tickers_explicit", "tickers_by_name"):
        val = row.get(col)
        result = first_elem(val)
        if result is not None:
            return result
    return None

df["primary_ticker"] = df.apply(primary_ticker, axis=1)

KEEP_COLS = [
    "id", "url", "date", "time", "datetime",
    "title", "sapo", "text", "body",
    "tickers_widget", "tickers_explicit", "tickers_by_name", "tickers_any",
    "n_tickers", "primary_ticker", "is_macro",
]
keep = [c for c in KEEP_COLS if c in df.columns]
df = df[keep].copy()

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_FILE, index=False, engine="pyarrow")
print(f"\n==> Written: {OUT_FILE}  ({len(df):,} rows)")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nRow counts after each step:")
for label, count in step_counts:
    print(f"  {label:40s}: {count:>8,}")

macro_rate = df["is_macro"].mean()
ticker_rate = 1.0 - macro_rate
print(f"\nMacro rate  : {macro_rate:.1%}")
print(f"Ticker rate : {ticker_rate:.1%}")

print("\nn_tickers distribution:")
vc = df["n_tickers"].value_counts().sort_index()
print(vc.to_string())

print("\nTop 20 primary_ticker:")
top20 = df["primary_ticker"].dropna().value_counts().head(20)
print(top20.to_string())

print("\n" + "=" * 60)
PASS = True
if len(df) < 15_000:
    print(f"FAIL: only {len(df):,} rows (requires >= 15,000)")
    PASS = False
else:
    print(f"PASS: {len(df):,} rows >= 15,000")

if ticker_rate < 0.30:
    print(f"FAIL: ticker rate = {ticker_rate:.1%} (requires >= 30%)")
    PASS = False
else:
    print(f"PASS: ticker rate = {ticker_rate:.1%} >= 30%")

DOCS_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(DOCS_FILE, "w", encoding="utf-8") as fh:
    fh.write("# Data Quality Report\n\n")
    fh.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    fh.write(f"**Source:** `data/raw/articles.jsonl`  \n")
    fh.write(f"**Output:** `data/interim/articles_clean.parquet`\n\n")
    fh.write("---\n\n")

    fh.write("## Row counts after each step\n\n")
    fh.write("| Step | Rows |\n")
    fh.write("|------|--------:|\n")
    for label, count in step_counts:
        fh.write(f"| {label} | {count:,} |\n")
    fh.write("\n")

    fh.write("## Macro / ticker rate\n\n")
    fh.write("| Metric | Value |\n")
    fh.write("|--------|--------:|\n")
    fh.write(f"| Macro rate | {macro_rate:.2%} |\n")
    fh.write(f"| Ticker rate | {ticker_rate:.2%} |\n")
    fh.write("\n")

    fh.write("## n_tickers distribution\n\n")
    fh.write("| n_tickers | Count |\n")
    fh.write("|----------:|-------:|\n")
    for idx, cnt in vc.items():
        fh.write(f"| {idx} | {cnt:,} |\n")
    fh.write("\n")

    fh.write("## Top 20 primary_ticker\n\n")
    fh.write("| Ticker | Count |\n")
    fh.write("|--------|-------:|\n")
    for ticker, cnt in top20.items():
        fh.write(f"| {ticker} | {cnt:,} |\n")
    fh.write("\n")

    status = "PASS" if PASS else "FAIL"
    fh.write(f"## Check result\n\n**{status}**\n\n")
    fh.write(f"- Output rows: **{len(df):,}** (requires >= 15,000)\n")
    fh.write(f"- Ticker rate: **{ticker_rate:.2%}** (requires >= 30%)\n")

print(f"\nReport: {DOCS_FILE}")
print("=" * 60)

if not PASS:
    sys.exit(1)
