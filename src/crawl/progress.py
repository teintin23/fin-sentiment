import json
from pathlib import Path

OUTPUT = Path("data/raw/articles.jsonl")

total = has_ticker = 0
if OUTPUT.exists():
    with OUTPUT.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            total += 1
            if rec.get("tickers_any"):
                has_ticker += 1

pct = has_ticker / total * 100 if total else 0.0
print(f"articles.jsonl  : {total:>7,} rows")
print(f"with tickers    : {has_ticker:>7,} ({pct:.1f}%)")
print(f"without ticker  : {total - has_ticker:>7,}")
if total >= 18000:
    print("PASS: >= 18000 rows")
else:
    print(f"Need            : {max(0, 18000 - total):>7,} more rows to PASS")
