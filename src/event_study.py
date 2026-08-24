"""
event_study.py
--------------
Event study theo Brown & Warner (1985) / MacKinlay (1997): kiem tra xem nhan
sentiment cua bai bao co di kem loi suat bat thuong (abnormal return) quanh
ngay dang tin hay khong.

Phuong phap:
- Mo hinh thi truong 1 nhan to: R_i = a + b*R_m + e, uoc luong OLS tren cua so
  t-130 .. t-11 (toi thieu 60 quan sat). AR = R_i - (a + b*R_m).
- Bien the --model market_adjusted: AR = R_i - R_m, khong hoi quy.
- Ngay su kien t0: tin dang sau 15:00 hoac vao ngay nghi -> phien giao dich
  ke tiep.
- Kiem dinh t cat ngang (cross-sectional) tren CAR tung nhom nhan; Welch t
  cho chenh lech POSITIVE - NEGATIVE.
- Cua so gia duoc [-5,-1] nam hoan toan truoc tin: neu khac 0 co y nghia thi
  ket qua dang bi nhiem (tin chong lan / ro ri truoc).

Dau ra: docs/event_study.md, reports/figures/event_study_caar.png

Usage:
    python src/event_study.py
    python src/event_study.py --model market_adjusted
    python src/event_study.py --source human
    python src/event_study.py --selftest      # kiem chung bang gia gia lap
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
PRICES = ROOT / "data" / "prices"
DOCS = ROOT / "docs"
FIGS = ROOT / "reports" / "figures"

LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
EST_START, EST_END = -130, -11        # cua so uoc luong (phien)
MIN_EST = 60                          # so quan sat toi thieu de hoi quy
WINDOWS = [(0, 0), (0, 1), (0, 3), (0, 5), (-1, 1)]
PLACEBO = (-5, -1)
PLOT_RANGE = (-10, 10)
NEWS_CUTOFF_HOUR = 15                 # tin sau gio nay tinh vao phien sau


# ------------------------------------------------------------- du lieu ----
def load_prices(price_dir: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Ma tran gia dong cua (ngay x ma) tren lich giao dich cua VNINDEX."""
    idx_path = price_dir / "VNINDEX.csv"
    if not idx_path.exists():
        sys.exit("Thieu data/prices/VNINDEX.csv - chay src/fetch_prices.py truoc.")
    cal = pd.to_datetime(pd.read_csv(idx_path)["time"]).sort_values()
    cal = pd.DatetimeIndex(cal.unique())

    cols = {}
    for p in sorted(price_dir.glob("*.csv")):
        if p.name.startswith("_"):
            continue
        d = pd.read_csv(p)
        s = pd.Series(pd.to_numeric(d["close"], errors="coerce").values,
                      index=pd.to_datetime(d["time"]))
        cols[p.stem] = s[~s.index.duplicated()].reindex(cal)
    px = pd.DataFrame(cols, index=cal)
    return px, cal


def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff()


def map_t0(dt: pd.Timestamp, cal: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Ngay su kien: phien giao dich dau tien ma tin co the tac dong."""
    d = dt.normalize()
    if dt.hour >= NEWS_CUTOFF_HOUR:
        d = d + pd.Timedelta(days=1)
    pos = cal.searchsorted(d)
    return cal[pos] if pos < len(cal) else None


def build_events(full: pd.DataFrame, ret: pd.DataFrame,
                 cal: pd.DatetimeIndex, source: str) -> pd.DataFrame:
    if source == "human":
        full = full[full["source_detail"] == "human"]
    elif source == "manual":
        full = full[full["source_detail"].isin(["human", "claude_manual"])]

    full = full[full["primary_ticker"].isin(ret.columns)]
    rows = []
    for _, r in full.iterrows():
        t0 = map_t0(r["datetime"], cal)
        if t0 is None:
            continue
        rows.append((r["primary_ticker"], t0, r["label"]))
    ev = pd.DataFrame(rows, columns=["ticker", "t0", "label"])

    # gop: cung ma + cung phien -> 1 su kien; nhan xung dot thi bo
    def agg(g: pd.DataFrame):
        labs = set(g["label"])
        return None if len(labs) > 1 else labs.pop()

    ev = (ev.groupby(["ticker", "t0"])["label"]
            .apply(lambda g: agg(pd.DataFrame({"label": g})))
            .reset_index())
    dropped = ev["label"].isna().sum()
    ev = ev.dropna(subset=["label"]).reset_index(drop=True)
    ev.attrs["dropped_conflict"] = int(dropped)
    return ev


# ---------------------------------------------------------------- loi ----
def compute_ars(ev: pd.DataFrame, ret: pd.DataFrame, rm: pd.Series,
                cal: pd.DatetimeIndex, model: str
                ) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """AR tung su kien tren luoi phien PLOT_RANGE. Tra ve (events, ar_matrix, stats)."""
    lo, hi = PLOT_RANGE
    days = np.arange(lo, hi + 1)
    pos_of = {d: i for i, d in enumerate(cal)}
    keep, mats = [], []
    n_short_est = 0

    for i, r in ev.iterrows():
        ri = ret[r["ticker"]]
        p0 = pos_of[r["t0"]]

        # uoc luong
        e_lo, e_hi = p0 + EST_START, p0 + EST_END
        if e_lo < 0:
            n_short_est += 1
            continue
        y = ri.iloc[e_lo:e_hi + 1]
        x = rm.iloc[e_lo:e_hi + 1]
        m = y.notna() & x.notna()
        if m.sum() < MIN_EST:
            n_short_est += 1
            continue
        if model == "market_model":
            b, a_ = np.polyfit(x[m].values, y[m].values, 1)
        else:  # market_adjusted
            a_, b = 0.0, 1.0

        # AR tren luoi phien
        row = np.full(len(days), np.nan)
        for j, d in enumerate(days):
            p = p0 + d
            if 0 <= p < len(cal):
                yi, xi = ri.iloc[p], rm.iloc[p]
                if np.isfinite(yi) and np.isfinite(xi):
                    row[j] = yi - (a_ + b * xi)
        if np.isnan(row[days.tolist().index(0)]):
            continue  # thieu chinh phien t0 thi bo
        keep.append(i)
        mats.append(row)

    stats = {"n_short_est": n_short_est}
    return ev.loc[keep].reset_index(drop=True), np.vstack(mats), stats


def car(ar: np.ndarray, w: tuple[int, int]) -> np.ndarray:
    lo, hi = PLOT_RANGE
    a, b = w
    seg = ar[:, a - lo:b - lo + 1]
    out = seg.sum(axis=1)
    out[np.isnan(seg).any(axis=1)] = np.nan
    return out


def tstat(x: np.ndarray) -> tuple[float, float, float, int]:
    """(mean, t, p, n) cat ngang."""
    from scipy import stats as st
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, n
    t, p = st.ttest_1samp(x, 0.0)
    return float(x.mean()), float(t), float(p), n


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int, int]:
    from scipy import stats as st
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan, len(a), len(b)
    t, p = st.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(t), float(p), len(a), len(b)


def overlap_share(ev: pd.DataFrame, cal: pd.DatetimeIndex, gap: int = 10) -> float:
    """Ti le su kien co su kien khac cung ma trong vong +/-gap phien."""
    pos_of = {d: i for i, d in enumerate(cal)}
    n_ov = 0
    for tk, g in ev.groupby("ticker"):
        ps = sorted(pos_of[t] for t in g["t0"])
        for i, p in enumerate(ps):
            near = (i > 0 and p - ps[i - 1] <= gap) or \
                   (i < len(ps) - 1 and ps[i + 1] - p <= gap)
            n_ov += near
    return n_ov / len(ev) if len(ev) else np.nan


# --------------------------------------------------------------- bao cao --
def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "-"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def run(price_dir: Path, out_md: Path, out_png: Path,
        model: str, source: str) -> dict:
    full = pd.read_parquet(PROC / "dataset_full.parquet")
    px, cal = load_prices(price_dir)
    ret = log_returns(px)
    if "VNINDEX" not in ret.columns:
        sys.exit("Thieu VNINDEX trong du lieu gia.")
    rm = ret["VNINDEX"]
    ret = ret.drop(columns=["VNINDEX"])

    ev0 = build_events(full, ret, cal, source)
    ev, ar, st_ = compute_ars(ev0, ret, rm, cal, model)
    if len(ev) < 30:
        print(f"CANH BAO: chi {len(ev)} su kien dung duoc - qua it de ket luan.")

    ov = overlap_share(ev, cal)
    lab_mask = {l: (ev["label"] == l).values for l in LABELS}

    # bang CAAR theo nhom x cua so
    rows_grp = []
    for w in WINDOWS + [PLACEBO]:
        c = car(ar, w)
        for l in LABELS:
            mu, t, p, n = tstat(c[lab_mask[l]])
            rows_grp.append({"cua_so": f"[{w[0]},{w[1]}]", "nhom": l,
                             "n": n, "CAAR_%": mu * 100 if np.isfinite(mu) else np.nan,
                             "t": t, "p": p})
    grp = pd.DataFrame(rows_grp)

    # chenh lech POS - NEG
    rows_diff = []
    for w in WINDOWS + [PLACEBO]:
        c = car(ar, w)
        d, t, p, na, nb = welch(c[lab_mask["POSITIVE"]], c[lab_mask["NEGATIVE"]])
        rows_diff.append({"cua_so": f"[{w[0]},{w[1]}]",
                          "POS-NEG_%": d * 100 if np.isfinite(d) else np.nan,
                          "t": t, "p": p, "n_POS": na, "n_NEG": nb})
    diff = pd.DataFrame(rows_diff)

    # placebo rieng
    plc = grp[grp["cua_so"] == f"[{PLACEBO[0]},{PLACEBO[1]}]"].copy()
    plc_diff = diff[diff["cua_so"] == f"[{PLACEBO[0]},{PLACEBO[1]}]"].iloc[0]

    # bieu do CAAR theo ngay
    plot_caar(ar, lab_mask, out_png)

    # ---- markdown ----
    w_ = []
    a = w_.append
    a("# Event study — sentiment và lợi suất bất thường\n")
    a(f"Mô hình: `{model}` | Nguồn nhãn: `{source}` | "
      f"Cửa sổ ước lượng: [{EST_START},{EST_END}] phiên, tối thiểu {MIN_EST} quan sát\n")

    a("## 1. Mẫu sự kiện\n")
    a("| | |")
    a("|---|---|")
    a(f"| Bài gốc khớp mã có giá | {len(ev0) + ev0.attrs['dropped_conflict']:,} sự kiện (đã gộp cùng mã cùng phiên) |")
    a(f"| Bỏ vì nhãn xung đột cùng phiên | {ev0.attrs['dropped_conflict']} |")
    a(f"| Bỏ vì thiếu dữ liệu ước lượng | {st_['n_short_est']} |")
    a(f"| **Sự kiện dùng được** | **{len(ev):,}** |")
    for l in LABELS:
        a(f"| — {l} | {int(lab_mask[l].sum()):,} |")
    a("")

    a("## 2. Phương pháp\n")
    a("Brown & Warner (1985), MacKinlay (1997). "
      + ("Mô hình thị trường một nhân tố ước lượng OLS trên cửa sổ "
         f"[{EST_START},{EST_END}] phiên trước sự kiện, AR = R − (α + βR_m)."
         if model == "market_model" else
         "Market-adjusted: AR = R − R_m, không hồi quy.")
      + " Tin đăng sau 15:00 hoặc ngày nghỉ tính vào phiên kế tiếp. "
        "Kiểm định t cắt ngang trên CAR.\n")

    a("## 3. CAAR theo nhóm nhãn\n")
    a("| Cửa sổ | Nhóm | n | CAAR % | t | p |")
    a("|---|---|---|---|---|---|")
    for _, r in grp[grp["cua_so"] != f"[{PLACEBO[0]},{PLACEBO[1]}]"].iterrows():
        a(f"| {r['cua_so']} | {r['nhom']} | {r['n']} | {r['CAAR_%']:+.3f} "
          f"| {r['t']:.2f} | {fmt_p(r['p'])} |")
    a("")

    a("## 4. Chênh lệch POSITIVE − NEGATIVE\n")
    a("Đây là bảng chính: nếu nhãn có giá trị thông tin thì chênh lệch phải "
      "dương ở các cửa sổ chứa t=0.\n")
    a("| Cửa sổ | POS−NEG % | t (Welch) | p | n POS | n NEG |")
    a("|---|---|---|---|---|---|")
    for _, r in diff[diff["cua_so"] != f"[{PLACEBO[0]},{PLACEBO[1]}]"].iterrows():
        a(f"| {r['cua_so']} | {r['POS-NEG_%']:+.3f} | {r['t']:.2f} "
          f"| {fmt_p(r['p'])} | {r['n_POS']} | {r['n_NEG']} |")
    a("")

    a("## 5. CAAR theo ngày\n")
    try:
        rel = out_png.relative_to(ROOT).as_posix()
    except ValueError:
        rel = out_png.name
    a(f"![CAAR]({rel})\n")

    a("## 6. Kiểm tra giả dược — cửa sổ [-5,-1]\n")
    a("Cửa sổ nằm hoàn toàn trước ngày tin, lẽ ra phải bằng 0. Khác 0 có ý "
      "nghĩa nghĩa là kết quả bị nhiễm (sự kiện chồng lấn hoặc thị trường "
      "phản ứng trước).\n")
    a("| Nhóm | n | CAAR % | t | p |")
    a("|---|---|---|---|---|")
    for _, r in plc.iterrows():
        a(f"| {r['nhom']} | {r['n']} | {r['CAAR_%']:+.3f} | {r['t']:.2f} "
          f"| {fmt_p(r['p'])} |")
    a(f"| POS−NEG | {plc_diff['n_POS']}/{plc_diff['n_NEG']} "
      f"| {plc_diff['POS-NEG_%']:+.3f} | {plc_diff['t']:.2f} "
      f"| {fmt_p(plc_diff['p'])} |")
    a("")

    a("## 7. Chồng lấn sự kiện\n")
    a(f"**{ov*100:.0f}%** sự kiện có sự kiện khác cùng mã trong vòng ±10 "
      "phiên. AR của tin trước tràn vào cửa sổ của tin sau, các quan sát "
      "không độc lập.\n")

    a("## 8. Hạn chế và cách đọc p-value\n")
    a("- Vì chồng lấn ở mục 7, **t-statistic bị thổi phồng, p-value thật lớn "
      "hơn con số in ra**. Đừng đọc `p < 0.05` ở đây như một thí nghiệm sạch.")
    a("- Nếu kết quả không có ý nghĩa, bốn khả năng chưa loại trừ được:")
    a("  1. Nhãn quá nhiễu (kappa 0.55, đa số là nhãn lan truyền)")
    a("  2. Gán mã sai ~15%")
    a("  3. Thị trường phản ứng trước khi tin lên báo")
    a("  4. Cỡ mẫu chưa đủ, nhất là nhóm NEGATIVE")
    a("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(w_) + "\n", encoding="utf-8")
    print(f"Ghi {out_md} va {out_png}")

    # in nhanh ra console
    key = diff[diff["cua_so"] == "[0,5]"].iloc[0]
    print(f"POS-NEG CAR[0,5] = {key['POS-NEG_%']:+.3f}%  "
          f"t={key['t']:.2f}  p={fmt_p(key['p'])}  (doc muc 6 va 8 truoc khi tin)")
    return {"grp": grp, "diff": diff, "n_events": len(ev), "overlap": ov}


def plot_caar(ar: np.ndarray, lab_mask: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = PLOT_RANGE
    days = np.arange(lo, hi + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"POSITIVE": "#2a9d3a", "NEUTRAL": "#888888", "NEGATIVE": "#c0392b"}
    for l in LABELS:
        m = np.nanmean(ar[lab_mask[l]], axis=0)
        ax.plot(days, np.nancumsum(m) * 100, label=f"{l} (n={lab_mask[l].sum()})",
                color=colors[l], lw=1.8)
    ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Phiên so với ngày tin (t=0)")
    ax.set_ylabel("CAAR (%)")
    ax.set_title("CAAR quanh ngày đăng tin, theo nhãn sentiment")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------- selftest --
def selftest(model: str) -> int:
    """Gia lap gia co cay tin hieu: POSITIVE +1.2%, NEGATIVE -1.5% tai t=0.
    Neu pipeline dung, CAAR[0,0] phai thu lai duoc xap xi cac con so nay."""
    import tempfile

    rng = np.random.default_rng(7)
    full = pd.read_parquet(PROC / "dataset_full.parquet")
    vc = full["primary_ticker"].value_counts()
    tickers = vc[vc >= 5].index.tolist()

    start = full["date"].min() - pd.Timedelta(days=280)
    end = full["date"].max() + pd.Timedelta(days=25)
    cal = pd.bdate_range(start, end)

    rm = rng.normal(0.0003, 0.009, len(cal))
    px = {"VNINDEX": 1000 * np.exp(np.cumsum(rm))}
    rets = {}
    for tk in tickers:
        r = 0.9 * rm + rng.normal(0, 0.012, len(cal))
        rets[tk] = r

    # cay tin hieu tai t0 cua tung su kien that
    plant = {"POSITIVE": 0.012, "NEGATIVE": -0.015, "NEUTRAL": 0.0}
    cal_idx = pd.DatetimeIndex(cal)
    pos_of = {d: i for i, d in enumerate(cal_idx)}
    for _, r in full[full["primary_ticker"].isin(tickers)].iterrows():
        t0 = map_t0(r["datetime"], cal_idx)
        if t0 is None or t0 not in pos_of:
            continue
        rets[r["primary_ticker"]][pos_of[t0]] += plant[r["label"]]

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        pdir = tdp / "prices"
        pdir.mkdir()
        pd.DataFrame({"time": cal.strftime("%Y-%m-%d"),
                      "close": px["VNINDEX"]}).to_csv(pdir / "VNINDEX.csv", index=False)
        for tk, r in rets.items():
            pd.DataFrame({"time": cal.strftime("%Y-%m-%d"),
                          "close": 50 * np.exp(np.cumsum(r))}
                         ).to_csv(pdir / f"{tk}.csv", index=False)

        res = run(pdir, tdp / "event_study.md", tdp / "caar.png",
                  model=model, source="all")

    g = res["grp"]
    got = {l: g[(g["cua_so"] == "[0,0]") & (g["nhom"] == l)]["CAAR_%"].iloc[0]
           for l in LABELS}
    print("\nSELFTEST — tin hieu cay vs thu lai (CAAR[0,0] %):")
    ok = True
    for l, want in [("POSITIVE", 1.2), ("NEGATIVE", -1.5), ("NEUTRAL", 0.0)]:
        tol = 0.35
        good = abs(got[l] - want) < tol
        ok &= good
        print(f"  {l:9s} cay {want:+.1f}  thu {got[l]:+.3f}  "
              f"{'OK' if good else 'LECH'}")
    print(f"  chong lan: {res['overlap']*100:.0f}% (ky vong cao — xem muc 7/8)")
    print("PASS" if ok else "LOI")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["market_model", "market_adjusted"],
                    default="market_model")
    ap.add_argument("--source", choices=["all", "manual", "human"], default="all")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.model)

    parts = []
    if args.model != "market_model":
        parts.append(args.model)
    if args.source != "all":
        parts.append(args.source)
    suffix = "_" + "_".join(parts) if parts else ""
    run(PRICES,
        DOCS / f"event_study{suffix}.md",
        FIGS / f"event_study_caar{suffix}.png",
        model=args.model, source=args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())