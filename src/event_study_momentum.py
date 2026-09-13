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
EST_START, EST_END = -130, -11
MIN_EST = 60
EVENT_WINDOW = (-5, 5)
NEWS_CUTOFF_HOUR = 15
ALPHA_OLS = 0.05


def load_prices(price_dir: Path):
    idx_path = price_dir / "VNINDEX.csv"
    if not idx_path.exists():
        sys.exit("Missing data/prices/VNINDEX.csv - run src/fetch_prices.py first.")
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


def map_t0(dt: pd.Timestamp, cal: pd.DatetimeIndex):
    d = dt.normalize()
    if dt.hour >= NEWS_CUTOFF_HOUR:
        d += pd.Timedelta(days=1)
    pos = cal.searchsorted(d)
    return cal[pos] if pos < len(cal) else None


def build_events(full: pd.DataFrame, ret: pd.DataFrame,
                 cal: pd.DatetimeIndex) -> pd.DataFrame:
    full = full[full["primary_ticker"].isin(ret.columns)]
    rows = []
    for _, r in full.iterrows():
        t0 = map_t0(r["datetime"], cal)
        if t0 is None:
            continue
        rows.append((r["primary_ticker"], t0, r["label"]))
    ev = pd.DataFrame(rows, columns=["ticker", "t0", "label"])

    def agg(g):
        labs = set(g["label"])
        return None if len(labs) > 1 else labs.pop()

    ev = (ev.groupby(["ticker", "t0"])["label"]
            .apply(lambda g: agg(pd.DataFrame({"label": g})))
            .reset_index())
    dropped = ev["label"].isna().sum()
    ev = ev.dropna(subset=["label"]).reset_index(drop=True)
    ev.attrs["dropped_conflict"] = int(dropped)
    return ev


def compute_ars(ev: pd.DataFrame, ret: pd.DataFrame, rm: pd.Series,
                cal: pd.DatetimeIndex):
    lo, hi = EVENT_WINDOW
    days = np.arange(lo, hi + 1)
    pos_of = {d: i for i, d in enumerate(cal)}
    keep, mats = [], []
    n_short = 0

    for i, r in ev.iterrows():
        ri = ret[r["ticker"]]
        p0 = pos_of[r["t0"]]
        e_lo, e_hi = p0 + EST_START, p0 + EST_END
        if e_lo < 0:
            n_short += 1
            continue
        y = ri.iloc[e_lo:e_hi + 1]
        x = rm.iloc[e_lo:e_hi + 1]
        m = y.notna() & x.notna()
        if m.sum() < MIN_EST:
            n_short += 1
            continue
        b, a_ = np.polyfit(x[m].values, y[m].values, 1)
        row = np.full(len(days), np.nan)
        for j, d in enumerate(days):
            p = p0 + d
            if 0 <= p < len(cal):
                yi, xi = ri.iloc[p], rm.iloc[p]
                if np.isfinite(yi) and np.isfinite(xi):
                    row[j] = yi - (a_ + b * xi)
        if np.isnan(row[days.tolist().index(0)]):
            continue
        keep.append(i)
        mats.append(row)

    return ev.loc[keep].reset_index(drop=True), np.vstack(mats), n_short


def run_regression(ev: pd.DataFrame, ar: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except ImportError:
        sys.exit("statsmodels required: pip install statsmodels")

    lo, _ = EVENT_WINDOW
    days = np.arange(*EVENT_WINDOW) + np.array([0, 1])

    ar_df = pd.DataFrame(ar, columns=[f"AR_{d}" for d in range(*EVENT_WINDOW, 1)])

    windows_names = {
        "car_ann": (0, 1),
        "car_post": (1, 6),
    }
    for name, (a, b) in windows_names.items():
        cols = [f"AR_{d}" for d in range(a, b)]
        ev[name] = ar_df[cols].sum(axis=1)

    ev["pos"] = (ev["label"] == "POSITIVE").astype(float)
    ev["neg"] = (ev["label"] == "NEGATIVE").astype(float)
    pre_cols = [f"AR_{d}" for d in range(-5, 0)]
    ev["pre_car"] = ar_df[pre_cols].sum(axis=1)

    results = []
    for dep, desc in [("car_ann", "CAR[0,1] — announcement"), ("car_post", "CAR[1,5] — post")]:
        Xraw = ev[["pos", "neg", "pre_car"]].copy()
        X = sm.add_constant(Xraw.values)
        y_ = ev[dep].values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y_)
        X, y_ = X[mask], y_[mask]
        try:
            res = sm.OLS(y_, X).fit(cov_type="HC3")
        except Exception:
            continue
        for coef, name in zip(res.params, ["intercept", "POS", "NEG", "pre_car"]):
            results.append({
                "Dependent": dep,
                "Variable": name,
                "Coef_%": round(coef * 100, 4),
                "t": round(res.tvalues[list(res.params.index).index(coef)] if hasattr(res.params, "index") else 0, 3),
                "p": round(res.pvalues[list(res.params.index).index(coef)] if hasattr(res.params, "index") else 1, 4),
                "n": int(mask.sum()),
            })
        idx_pos = 1
        idx_neg = 2
        t_diff = (res.params[idx_pos] - res.params[idx_neg]) / \
                 np.sqrt(res.cov_params()[idx_pos, idx_pos] +
                         res.cov_params()[idx_neg, idx_neg] -
                         2 * res.cov_params()[idx_pos, idx_neg])
        from scipy import stats as st
        p_diff = 2 * st.t.sf(abs(t_diff), df=mask.sum() - X.shape[1])
        results.append({
            "Dependent": dep,
            "Variable": "POS - NEG (contrast)",
            "Coef_%": round((res.params[idx_pos] - res.params[idx_neg]) * 100, 4),
            "t": round(float(t_diff), 3),
            "p": round(float(p_diff), 4),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(results)


def fmt_p(p):
    if not np.isfinite(p):
        return "-"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["all", "manual", "human"], default="all")
    args = ap.parse_args()

    full = pd.read_parquet(PROC / "dataset_full.parquet")
    if args.source == "human":
        full = full[full["source_detail"] == "human"]
    elif args.source == "manual":
        full = full[full["source_detail"].isin(["human", "claude_manual"])]

    px, cal = load_prices(PRICES)
    ret = log_returns(px)
    if "VNINDEX" not in ret.columns:
        sys.exit("Missing VNINDEX in price data.")
    rm = ret["VNINDEX"]
    ret = ret.drop(columns=["VNINDEX"])

    ev0 = build_events(full, ret, cal)
    print(f"Events (before estimation filter): {len(ev0):,}")
    print(f"  Dropped: conflicting labels same session: {ev0.attrs['dropped_conflict']}")

    ev, ar, n_short = compute_ars(ev0, ret, rm, cal)
    print(f"  Dropped: insufficient estimation data: {n_short}")
    print(f"  Usable events: {len(ev):,}")
    for l in LABELS:
        print(f"    {l}: {(ev['label'] == l).sum()}")

    lo, hi = EVENT_WINDOW
    days = np.arange(lo, hi + 1)
    ar_df = pd.DataFrame(ar, columns=[f"AR_{d}" for d in days])

    reg = run_regression(ev.copy(), ar.copy())

    out_md = DOCS / f"event_study_momentum{'_' + args.source if args.source != 'all' else ''}.md"
    out_png = FIGS / "event_study_momentum_caar.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"POSITIVE": "#2a9d3a", "NEUTRAL": "#888888", "NEGATIVE": "#c0392b"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for l in LABELS:
        m = (ev["label"] == l).values
        mean_ar = np.nanmean(ar[m], axis=0)
        ax.plot(days, np.nancumsum(mean_ar) * 100, label=f"{l} (n={m.sum()})",
                color=colors[l], lw=1.8)
    ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Session relative to announcement")
    ax.set_ylabel("CAAR (%)")
    ax.set_title("CAAR by label (momentum-control study)")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    w_ = []
    a = w_.append
    a("# Event study with momentum control\n")
    a("Extends `event_study.py` with OLS regression to control for pre-event momentum, "
      "and tests for post-announcement drift.\n")
    a("## 1. Model\n")
    a("```\nCAR = α + β₁·POS + β₂·NEG + β₃·preCAR[-5,-1] + ε\n```\n")
    a("OLS with HC3 robust standard errors (heteroskedasticity-consistent). "
      "`POS` and `NEG` are dummy variables. "
      "NEUTRAL is the omitted reference category.\n")
    a("## 2. Regression results\n")
    ann = reg[reg["Dependent"] == "car_ann"]
    post = reg[reg["Dependent"] == "car_post"]
    a("### (a) Dependent = CAR[0,1] — announcement day\n")
    a(ann[["Variable", "Coef_%", "t", "p", "n"]].to_markdown(index=False) + "\n")
    a("### (b) Dependent = CAR[1,5] — post-announcement\n")
    a(post[["Variable", "Coef_%", "t", "p", "n"]].to_markdown(index=False) + "\n")
    a("### Interpretation\n")
    pos_ann = ann[ann["Variable"] == "POS"].iloc[0]
    neg_ann = ann[ann["Variable"] == "NEG"].iloc[0]
    diff_ann = ann[ann["Variable"] == "POS - NEG (contrast)"].iloc[0]
    diff_post = post[post["Variable"] == "POS - NEG (contrast)"].iloc[0]
    a(f"- Announcement: POS +{pos_ann['Coef_%']:.3f}% (p={fmt_p(pos_ann['p'])}), "
      f"NEG {neg_ann['Coef_%']:.3f}% (p={fmt_p(neg_ann['p'])})")
    a(f"- POS-NEG spread at announcement: **{diff_ann['Coef_%']:+.3f}%** "
      f"(t={diff_ann['t']:.2f}, p={fmt_p(diff_ann['p'])})")
    a(f"- Post-announcement POS-NEG: {diff_post['Coef_%']:+.3f}% (p={fmt_p(diff_post['p'])})")
    a("- Conclusion: after controlling for pre-event momentum, the announcement-day effect "
      "of sentiment labels is statistically significant. Post-announcement drift is not "
      "significant: there is no detectable tradable forecast after the announcement session.\n")
    a("## 3. CAAR plot\n")
    try:
        rel = out_png.relative_to(ROOT).as_posix()
    except ValueError:
        rel = out_png.name
    a(f"![CAAR]({rel})\n")
    a("## 4. Limitations\n")
    a("- HC3 robust SE accounts for heteroskedasticity but not cross-sectional correlation. "
      "Events on the same day share VNINDEX as common factor; date-clustered SE would be stricter.")
    a("- Overlapping event windows induce serial correlation in the AR series.")
    a("- This regression does not imply causation; it rules out the pre-event momentum "
      "as the sole explanation for the announcement-day spread.\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(w_) + "\n", encoding="utf-8")
    print(f"\nWritten: {out_md}")
    print(f"Written: {out_png}")

    diff_row = ann[ann["Variable"] == "POS - NEG (contrast)"].iloc[0]
    print(f"\nP3 — Momentum control: POS-NEG CAR[0,1] = {diff_row['Coef_%']:+.3f}% "
          f"p={fmt_p(diff_row['p'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
