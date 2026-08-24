"""
fetch_prices.py
---------------
Tai gia dong cua theo ngay cho cac ma co >= --min-articles bai trong dataset,
cong them VNINDEX, qua vnstock 4.x (API moi, Quote().history()).

- Resume: file da co thi bo qua, ngat giua chung chay lai khong mat gi.
- Ma loi luu vao data/prices/_failed.json, chay --retry-failed de thu lai.
- Ba nguon du phong theo thu tu: kbs -> vci -> dnse. vnstock doi API lan nua
  thi con duong cuoi: tu viet fetch tho vao ham fetch_raw() ben duoi.

Usage:
    python src/fetch_prices.py
    python src/fetch_prices.py --retry-failed
    python src/fetch_prices.py --min-articles 3
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "prices"
FAILED = OUT / "_failed.json"

SOURCES = ["kbs", "vci", "dnse"]  # thu tu du phong

# vnstock chan theo phut: Guest 20 req/phut, co API key mien phi 60 req/phut.
# Vuot nguong la vnai TAT LUON tien trinh, nen phai tu ghim toc do duoi tran.
_req_times: list[float] = []


def throttle(rpm: int) -> None:
    """Giu so request trong 60s gan nhat < rpm."""
    now = time.monotonic()
    while _req_times and now - _req_times[0] > 61:
        _req_times.pop(0)
    if len(_req_times) >= rpm:
        wait = 61 - (now - _req_times[0]) + 0.5
        if wait > 0:
            time.sleep(wait)
    _req_times.append(time.monotonic())


# --------------------------------------------------------------- fetch ----
def fetch_raw(symbol: str, start: str, end: str, rpm: int) -> pd.DataFrame | None:
    """Thu lan luot tung nguon. Tra ve DataFrame co cot time, close hoac None.

    Bat ca SystemExit vi vnai goi sys.exit() khi dinh rate limit."""
    from vnstock import Quote

    last_err = None
    for src in SOURCES:
        for attempt in range(3):
            throttle(rpm)
            try:
                q = Quote(source=src, symbol=symbol, show_log=False)
                df = q.history(start=start, end=end, interval="1D")
                if df is None or len(df) == 0:
                    last_err = f"{src}: rong"
                    break  # nguon nay khong co du lieu, sang nguon khac
                df = normalize(df)
                if df is not None and len(df) >= 5:
                    return df
                last_err = f"{src}: khong nhan dang duoc cot"
                break
            except (Exception, SystemExit) as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                last_err = f"{src}: {msg}"
                if "rate" in msg.lower() or "limit" in msg.lower() \
                        or "gioi han" in msg.lower():
                    print(f"    rate limit tai {symbol}, cho 65s...")
                    time.sleep(65)
                    _req_times.clear()
                    continue  # thu lai cung nguon
                break  # loi khac -> sang nguon khac
    print(f"    LOI {symbol}: {last_err}")
    return None


def normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    """Dua ve dang chuan 2 cot: time (YYYY-MM-DD), close (float)."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in ("time", "date", "trading_date", "tradingdate")
                 if c in df.columns), None)
    ccol = next((c for c in ("close", "close_price", "closeprice", "adclose",
                             "adj_close") if c in df.columns), None)
    if tcol is None or ccol is None:
        return None
    out = df[[tcol, ccol]].rename(columns={tcol: "time", ccol: "close"})
    out["time"] = pd.to_datetime(out["time"]).dt.strftime("%Y-%m-%d")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna().drop_duplicates("time").sort_values("time")
    return out


# ---------------------------------------------------------------- main ----
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-articles", type=int, default=5)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="tai lai ca nhung file da co")
    ap.add_argument("--rpm", type=int, default=18,
                    help="so request toi da moi phut. Guest=20 (de 18 cho an toan), "
                         "co API key mien phi=60 (de 55)")
    args = ap.parse_args()

    full = pd.read_parquet(PROC / "dataset_full.parquet")
    vc = full["primary_ticker"].value_counts()
    tickers = sorted(vc[vc >= args.min_articles].index.tolist())

    # cua so gia: du 130 phien uoc luong truoc su kien dau + 10 phien sau su kien cuoi
    start = (full["date"].min() - timedelta(days=280)).strftime("%Y-%m-%d")
    end = (full["date"].max() + timedelta(days=25)).strftime("%Y-%m-%d")

    todo = ["VNINDEX"] + tickers
    if args.retry_failed and FAILED.exists():
        prev = json.loads(FAILED.read_text(encoding="utf-8"))
        todo = [t for t in todo if t in prev]
        print(f"Retry {len(todo)} ma tu _failed.json")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Tai {len(todo)} ma ({start} -> {end}), min-articles={args.min_articles}")

    failed: dict[str, str] = {}
    ok = skip = 0
    for i, sym in enumerate(todo, 1):
        path = OUT / f"{sym}.csv"
        if path.exists() and path.stat().st_size > 200 and not args.force:
            skip += 1
            continue
        df = fetch_raw(sym, start, end, args.rpm)
        if df is None:
            failed[sym] = "khong tai duoc tu ca 3 nguon"
        else:
            df.to_csv(path, index=False)
            ok += 1
        if i % 20 == 0:
            print(f"  {i}/{len(todo)}  ok={ok} skip={skip} loi={len(failed)}", flush=True)

    # gop failed cu (neu khong phai retry thi ghi de)
    if args.retry_failed and FAILED.exists():
        prev = json.loads(FAILED.read_text(encoding="utf-8"))
        for k in todo:
            prev.pop(k, None)
        prev.update(failed)
        failed = prev
    if failed:
        FAILED.write_text(json.dumps(failed, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    elif FAILED.exists():
        FAILED.unlink()

    have = sorted(p.stem for p in OUT.glob("*.csv"))
    has_idx = "VNINDEX" in have
    print()
    print(f"Tong file gia   : {len(have)}")
    print(f"co VNINDEX      : {'CO' if has_idx else 'KHONG'}")
    print(f"Loi con lai     : {len(failed)}"
          + ("  (chay --retry-failed)" if failed else ""))
    status = "PASS" if has_idx and len(have) >= 50 else "LOI"
    print(status)
    if not has_idx:
        print("Thieu VNINDEX -> event study khong chay duoc."
              " Neu bao loi lien tuc, co the can API key:"
              " vnstocks.com/login -> vnai.setup_api_key('key')")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())