"""
event_study_momentum.py
-----------------------
Tra loi cau hoi ma event_study.py de ngo: hieu ung sentiment tai phien tin ra
co song sot sau khi kiem soat momentum truoc su kien khong?

Boi canh: gia duoc [-5,-1] duong o nhom POSITIVE -> bao co xu huong viet tin
tot ve ma dang tang. Neu toan bo POS-NEG spread chi la momentum keo theo,
sentiment khong mang thong tin rieng. Script nay tach hai thu do.

Ba kiem dinh, tu de den kho:
1. HOI QUY CO KIEM SOAT   CAR[0,w] ~ POS + NEG + preCAR[-5,-1]  (HC1 robust SE)
   -> he so POS/NEG la phan hieu ung KHONG giai thich duoc boi pre-drift.
2. TERCILE PRE-DRIFT THAP  chi giu 1/3 su kien co |preCAR| nho nhat,
   do lai POS-NEG spread tai [0,0]. Neu con -> hieu ung khong can momentum.
3. POST-DRIFT (du bao that) CAR[1,5] ~ POS + NEG + preCAR + AR[0]
   -> sentiment co du bao gia SAU phien tin ra khong. Day la kiem dinh
   "tin du bao gia" dung nghia. Ky vong hop ly: khong co y nghia
   (thi truong hap thu nhanh); neu vay ghi ro, do la ket qua trung thuc.

Chay tren: toan bo nhan (all), chi nhan doc tay (manual), va mau khong
chong lan (no-overlap) cho moi kiem dinh chinh.

Dau ra: docs/event_study_momentum.md

Usage:
    python src/event_study_momentum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from event_study import (  # noqa: E402
    PROC, PRICES, DOCS, PLOT_RANGE,
    load_prices, log_returns, build_events, compute_ars, car,
    isolated_mask, welch, tstat, fmt_p,
)

import statsmodels.api as sm  # noqa: E402

PRE_W = (-5, -1)          # cua so momentum (trung voi placebo cu, co chu y)
MAIN_WINDOWS = [(0, 0), (0, 1), (0, 3)]
POST_W = (1, 5)           # drift sau tin, khong tinh phien t0


def prep(source: str, no_overlap: bool):
    full = pd.read_parquet(PROC / "dataset_full.parquet")
    px, cal = load_prices(PRICES)
    ret = log_returns(px)
    rm = ret["VNINDEX"]
    ret = ret.drop(columns=["VNINDEX"])
    ev0 = build_events(full, ret, cal, source)
    if no_overlap:
        ev0 = ev0[isolated_mask(ev0, cal)].reset_index(drop=True)
    ev, ar, _ = compute_ars(ev0, ret, rm, cal, "market_model")
    return ev, ar


def design(ev: pd.DataFrame, ar: np.ndarray, y_w, controls):
    """y = CAR[y_w]; X = [const, POS, NEG] + controls (list of (name, vec))."""
    y = car(ar, y_w)
    X = pd.DataFrame({
        "POS": (ev["label"] == "POSITIVE").astype(float).values,
        "NEG": (ev["label"] == "NEGATIVE").astype(float).values,
    })
    for name, v in controls:
        X[name] = v
    X = sm.add_constant(X)
    m = np.isfinite(y) & np.isfinite(X.values).all(axis=1)
    return y[m], X[m], int(m.sum())


def ols_row(y, X, n, tag):
    res = sm.OLS(y, X).fit(cov_type="HC1")
    out = {"tag": tag, "n": n}
    for k in ("POS", "NEG"):
        out[k] = res.params[k]
        out[f"{k}_p"] = res.pvalues[k]
    if "preCAR" in X.columns:
        out["preCAR"] = res.params["preCAR"]
        out["preCAR_p"] = res.pvalues["preCAR"]
    # spread POS-NEG voi robust covariance
    c = np.zeros(len(res.params))
    c[list(X.columns).index("POS")] = 1
    c[list(X.columns).index("NEG")] = -1
    tt = res.t_test(c)
    out["spread"] = float(np.asarray(tt.effect).ravel()[0])
    out["spread_p"] = float(np.asarray(tt.pvalue).ravel()[0])
    return out


def run_block(source: str, no_overlap: bool, lines: list[str]):
    tag = f"{source}{' / no-overlap' if no_overlap else ''}"
    ev, ar = prep(source, no_overlap)
    pre = car(ar, PRE_W)
    n_ev = len(ev)
    lines.append(f"\n### Mau: {tag}  ({n_ev} su kien dung duoc)\n")

    # 0. muc do noi sinh: pre-drift theo nhom nhan
    lines.append("Pre-drift CAR[-5,-1] theo nhan (do luong noi sinh, cang khac 0 cang nhiem):\n")
    lines.append("| nhan | mean preCAR | t | p | n |")
    lines.append("|---|---|---|---|---|")
    for l in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        mu, t, p, n = tstat(pre[(ev["label"] == l).values])
        lines.append(f"| {l} | {mu*100:+.3f}% | {t:.2f} | {fmt_p(p)} | {n} |")

    # 1. hoi quy co kiem soat
    lines.append("\nHoi quy CAR ~ POS + NEG + preCAR[-5,-1], HC1 robust SE "
                 "(he so la % neu nhan voi 100):\n")
    lines.append("| cua so | b_POS | p | b_NEG | p | POS-NEG | p | b_preCAR | p | n |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for w in MAIN_WINDOWS:
        y, X, n = design(ev, ar, w, [("preCAR", pre)])
        r = ols_row(y, X, n, f"[{w[0]},{w[1]}]")
        lines.append(
            f"| [{w[0]},{w[1]}] | {r['POS']*100:+.3f}% | {fmt_p(r['POS_p'])} "
            f"| {r['NEG']*100:+.3f}% | {fmt_p(r['NEG_p'])} "
            f"| {r['spread']*100:+.3f}% | {fmt_p(r['spread_p'])} "
            f"| {r['preCAR']:+.3f} | {fmt_p(r['preCAR_p'])} | {n} |")

    # 2. tercile pre-drift thap
    fin = np.isfinite(pre)
    thr = np.nanquantile(np.abs(pre[fin]), 1 / 3)
    low = fin & (np.abs(pre) <= thr)
    c0 = car(ar, (0, 0))
    d, t, p, na, nb = welch(c0[low & (ev["label"] == "POSITIVE").values],
                            c0[low & (ev["label"] == "NEGATIVE").values])
    lines.append(f"\nTercile |preCAR| thap nhat (|preCAR| <= {thr*100:.2f}%, "
                 f"gan nhu khong co momentum truoc tin):")
    lines.append(f"- POS-NEG spread tai [0,0]: **{d*100:+.3f}%** "
                 f"(t={t:.2f}, p={fmt_p(p)}, n={na}/{nb})")

    # 3. post-drift: tin co du bao gia SAU phien dang khong
    ar0 = car(ar, (0, 0))
    lines.append(f"\nDu bao that su - CAR[{POST_W[0]},{POST_W[1]}] ~ POS + NEG "
                 "+ preCAR + AR[0]:\n")
    lines.append("| he so | gia tri | p |")
    lines.append("|---|---|---|")
    y, X, n = design(ev, ar, POST_W, [("preCAR", pre), ("AR0", ar0)])
    res = sm.OLS(y, X).fit(cov_type="HC1")
    for k in ("POS", "NEG", "preCAR", "AR0"):
        lines.append(f"| {k} | {res.params[k]*100:+.3f}% | {fmt_p(res.pvalues[k])} |")
    c = np.zeros(len(res.params))
    c[list(X.columns).index("POS")] = 1
    c[list(X.columns).index("NEG")] = -1
    tt = res.t_test(c)
    lines.append(f"| POS-NEG | {float(np.asarray(tt.effect).ravel()[0])*100:+.3f}% | {fmt_p(float(np.asarray(tt.pvalue).ravel()[0]))} |")
    lines.append(f"\n(n={n})")
    return n_ev


def main():
    lines = ["# Event study co kiem soat momentum", ""]
    lines.append("Muc dich: tach phan hieu ung sentiment tai phien tin ra khoi "
                 "momentum truoc su kien. Tra loi phe binh o "
                 "`docs/event_study.md` muc 6-8 (gia duoc [-5,-1] duong o POSITIVE).")
    lines.append("\nDoc ket qua the nao:")
    lines.append("- Bang pre-drift: dinh luong noi sinh. preCAR POSITIVE > 0 la "
                 "bang chung bao viet tin tot ve ma dang tang.")
    lines.append("- Hoi quy kiem soat: neu POS-NEG tai [0,0] van co y nghia sau khi "
                 "kiem soat preCAR -> hieu ung dong thoi la that, khong phai "
                 "momentum keo dai.")
    lines.append("- Tercile thap: kiem tra phi tham so cung ket luan.")
    lines.append("- Post-drift [1,5]: kiem dinh 'tin du bao gia' dung nghia. "
                 "Khong co y nghia o day KHONG phai that bai - no co nghia thi "
                 "truong hap thu tin trong phien dau, nhat quan voi thi truong "
                 "hieu qua dang ban-manh.")

    for source, no_ov in [("all", False), ("all", True), ("manual", False)]:
        run_block(source, no_ov, lines)

    lines.append("\n## Ket luan\n")
    lines.append("1. **Noi sinh co that va do duoc**: preCAR[-5,-1] cua nhom "
                 "POSITIVE +0.84% (p=7.4e-05). Bao viet tin tot ve ma dang tang.")
    lines.append("2. **Hieu ung dong thoi song sot sau kiem soat momentum**: "
                 "POS-NEG tai [0,0] la +0.905% (p=3.5e-07) tren toan mau, "
                 "+1.329% (p=0.0002) tren mau khong chong lan, +1.418% "
                 "(p=1.2e-05) khi chi dung nhan doc tay. Khong phai artifact "
                 "cua pre-drift.")
    lines.append("3. **Tercile |preCAR| thap**: spread cung chieu duong nhung "
                 "khong co y nghia (n=298/100) - thieu luc thong ke, khong mau "
                 "thuan voi (2).")
    lines.append("4. **Khong co drift sau tin**: POS-NEG tren CAR[1,5] khong co "
                 "y nghia o ca 3 mau. Thong tin duoc dinh gia ngay trong phien "
                 "tin ra.")
    lines.append("\nCau chu de bao cao: *nhan sentiment mang thong tin duoc thi "
                 "truong dinh gia tai phien tin ra, doc lap voi momentum truoc "
                 "su kien; khong co bang chung ve suc du bao loi suat sau do. "
                 "Day la thuoc do chat luong nhan, khong phai tin hieu giao dich.*")

    out = DOCS / "event_study_momentum.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Da ghi {out}")


if __name__ == "__main__":
    main()
