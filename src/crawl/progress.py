"""progress.py – Xem tiến độ crawl đang chạy (an toàn với file đang ghi)."""
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
print(f"articles.jsonl  : {total:>7,} dòng")
print(f"có tickers_any  : {has_ticker:>7,} bài  ({pct:.1f}%)")
print(f"chưa có ticker  : {total - has_ticker:>7,} bài")
if total >= 18000:
    print("PASS: >= 18000 dòng ✓")
else:
    print(f"Cần thêm        : {max(0, 18000 - total):>7,} dòng để PASS")
