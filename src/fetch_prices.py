from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
WATCHLIST = ROOT / "config" / "watchlist.txt"
TICKERS_TXT = ROOT / "data" / "tickers.txt"
OUT_DIR = ROOT / "data" / "prices"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; vn-fin-sentiment/1.0)"
}

MAX_RETRY = 3
RETRY_SLEEP = 5
RPM_DEFAULT = 60


def load_tickers(watchlist: bool) -> list[str]:
    if watchlist:
        if not WATCHLIST.exists():
            sys.exit(f"Watchlist not found: {WATCHLIST}")
        raw = WATCHLIST.read_text(encoding="utf-8")
    else:
        if not TICKERS_TXT.exists():
            sys.exit(f"Tickers not found: {TICKERS_TXT}")
        raw = TICKERS_TXT.read_text(encoding="utf-8")
    return [t.strip().upper() for t in raw.splitlines()
            if t.strip() and not t.strip().startswith("#")]


class Throttle:
    def __init__(self, rpm: int) -> None:
        self._min_gap = 60.0 / max(rpm, 1)
        self._history: list[float] = []

    def wait(self) -> None:
        now = time.time()
        self._history = [t for t in self._history if now - t < 60]
        if len(self._history) >= 60:
            time.sleep(max(0, 60 - (now - self._history[0])))
        elif self._history:
            gap = now - self._history[-1]
            if gap < self._min_gap:
                time.sleep(self._min_gap - gap)
        self._history.append(time.time())


def fetch_raw(ticker: str, throttle: Throttle) -> pd.DataFrame | None:
    for src in ("CAFEF", "TCBS", "VCI"):
        throttle.wait()
        try:
            from vnstock import Stock
            s = Stock(symbol=ticker, source=src)
            df = s.quote.history(start="2023-01-01", end=None, interval="1D")
            if df is not None and len(df) >= 100:
                return df
        except Exception:
            pass
    return None


def normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    time_col = next((c for c in ("time", "date", "Date", "Time") if c in df.columns), None)
    close_col = next((c for c in ("close", "Close", "closePrice", "close_price")
                      if c in df.columns), None)
    if time_col is None or close_col is None:
        return None
    out = pd.DataFrame({"time": pd.to_datetime(df[time_col], errors="coerce"),
                        "close": pd.to_numeric(df[close_col], errors="coerce")})
    out = out.dropna()
    out = out.sort_values("time").drop_duplicates(subset="time")
    if len(out) < 100:
        return None
    out["time"] = out["time"].dt.strftime("%Y-%m-%d")
    return out


def fetch_ticker(ticker: str, throttle: Throttle, force: bool) -> bool:
    out = OUT_DIR / f"{ticker}.csv"
    if out.exists() and not force:
        return True

    for attempt in range(1, MAX_RETRY + 1):
        df = fetch_raw(ticker, throttle)
        if df is None:
            if attempt < MAX_RETRY:
                time.sleep(RETRY_SLEEP * attempt)
            continue
        norm = normalize(df)
        if norm is None:
            break
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        norm.to_csv(out, index=False)
        return True

    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", action="store_true",
                    help="fetch watchlist tickers only instead of all tickers")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if file already exists")
    ap.add_argument("--rpm", type=int, default=RPM_DEFAULT,
                    help=f"max requests per minute (default: {RPM_DEFAULT})")
    args = ap.parse_args()

    tickers = load_tickers(args.watchlist)
    tickers = sorted(set(tickers))
    if "VNINDEX" not in tickers:
        tickers = ["VNINDEX"] + tickers

    throttle = Throttle(args.rpm)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Fetching prices — {len(tickers)} tickers, max {args.rpm} rpm")
    print("=" * 60)

    n_ok = n_fail = 0
    failed: list[str] = []
    for i, tk in enumerate(tickers, 1):
        ok = fetch_ticker(tk, throttle, args.force)
        if ok:
            n_ok += 1
            print(f"  [{i:3d}/{len(tickers)}]  OK      {tk}")
        else:
            n_fail += 1
            failed.append(tk)
            print(f"  [{i:3d}/{len(tickers)}]  FAILED  {tk}")

    print()
    print("=" * 60)
    print(f"Success: {n_ok} | Failed: {n_fail}")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")
    if "VNINDEX" in failed:
        print("ERROR: VNINDEX failed — required for event study.")
        return 1
    print("Done. Run: python src/event_study.py")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())