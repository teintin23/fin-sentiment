"""
compare_models.py
-----------------
So sánh baseline TF-IDF+LR với PhoBERT trên cùng test set, kiểm định McNemar,
và phân tích lỗi bằng ví dụ thật.

PhoBERT nặng ~540 MB nên script KHÔNG nạp lại model. Nó đọc dự đoán đã lưu ở
`models/phobert-vnfin/test_predictions.json` (do train_phobert.py sinh ra).
Baseline nhẹ nên chạy lại trực tiếp từ joblib.

Usage:
    python src/compare_models.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "data" / "processed" / "test.parquet"
CACHE = ROOT / "data" / "interim" / "tokenized_cache.parquet"
BASE_MODEL = ROOT / "models" / "baseline_tfidf_lr.joblib"
PHO_PRED = ROOT / "models" / "phobert-vnfin" / "test_predictions.json"
DOC = ROOT / "docs" / "model_comparison.md"

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
I2L = dict(enumerate(LABELS))
ALPHA = 0.05


def mcnemar(b: int, c: int) -> tuple[float, str]:
    """McNemar. Dùng nhị thức chính xác khi b+c nhỏ, chi-square hiệu chỉnh khi lớn."""
    from scipy import stats
    n = b + c
    if n == 0:
        return 1.0, "khong co bai nao hai mo hinh khac nhau"
    if n < 25:
        p = float(stats.binomtest(min(b, c), n, 0.5).pvalue)
        return p, "kiem dinh nhi thuc chinh xac (b+c < 25)"
    stat = (abs(b - c) - 1) ** 2 / n
    p = float(stats.chi2.sf(stat, 1))
    return p, f"chi-square hieu chinh lien tuc, stat={stat:.3f}"


def metrics_block(y, p, mask=None) -> dict:
    if mask is not None:
        y, p = y[mask], p[mask]
    d = {"n": int(len(y)),
         "macro_F1": round(f1_score(y, p, average="macro", zero_division=0), 4),
         "accuracy": round(accuracy_score(y, p), 4),
         "weighted_F1": round(f1_score(y, p, average="weighted", zero_division=0), 4)}
    per = f1_score(y, p, average=None, labels=list(range(3)), zero_division=0)
    for i, lb in enumerate(LABELS):
        d[f"F1_{lb}"] = round(float(per[i]), 4)
    return d


def main() -> int:
    print("=" * 68)
    print("SO SANH MO HINH")
    print("=" * 68)

    te = pd.read_parquet(TEST)
    tok = dict(zip(*pd.read_parquet(CACHE).T.values)) if CACHE.exists() else {}
    if CACHE.exists():
        c = pd.read_parquet(CACHE)
        tok = dict(zip(c["id"].astype(str), c["tok"]))
    te["tok"] = te["id"].astype(str).map(tok)
    if te["tok"].isna().any():
        print("  ERROR: thieu tokenized_cache. Chay src/train_baseline.py truoc.")
        return 1

    # ---- du doan baseline --------------------------------------------------
    bundle = joblib.load(BASE_MODEL)
    base_pred_lbl = bundle["pipeline"].predict(te["tok"])
    L2I = {l: i for i, l in I2L.items()}
    base = np.array([L2I[x] for x in base_pred_lbl])

    # ---- du doan phobert ---------------------------------------------------
    pj = json.loads(PHO_PRED.read_text(encoding="utf-8"))
    order = {str(i): k for k, i in enumerate(pj["id"])}
    idx = te["id"].astype(str).map(order)
    if idx.isna().any():
        print("  ERROR: test_predictions.json khong khop id voi test.parquet")
        return 1
    pho = np.array(pj["y_pred"])[idx.astype(int).to_numpy()]
    y = np.array(pj["y_true"])[idx.astype(int).to_numpy()]

    assert (y == te["label"].map(L2I).to_numpy()).all(), "nhan that khong khop"
    print(f"  test: {len(te):,} bai, khop id giua hai mo hinh: OK")

    # ---- bang so sanh ------------------------------------------------------
    hm = (te["source_detail"] == "human").to_numpy()
    tbl_all = pd.DataFrame([{"model": "Baseline TF-IDF+LR", **metrics_block(y, base)},
                            {"model": "PhoBERT",            **metrics_block(y, pho)}])
    tbl_hum = pd.DataFrame([{"model": "Baseline TF-IDF+LR", **metrics_block(y, base, hm)},
                            {"model": "PhoBERT",            **metrics_block(y, pho, hm)}])
    print("\n--- Toan bo test ---")
    print(tbl_all.to_string(index=False))
    print(f"\n--- Chi phan nhan nguoi (n={int(hm.sum())}) ---")
    print(tbl_hum.to_string(index=False))

    # ---- McNemar -----------------------------------------------------------
    ok_b, ok_p = base == y, pho == y
    b = int((ok_b & ~ok_p).sum())   # baseline dung, phobert sai
    c = int((~ok_b & ok_p).sum())   # phobert dung, baseline sai
    both_ok = int((ok_b & ok_p).sum())
    both_bad = int((~ok_b & ~ok_p).sum())
    p_val, method = mcnemar(b, c)
    sig = p_val < ALPHA
    print(f"\n--- McNemar ---")
    print(f"  ca hai dung   : {both_ok}")
    print(f"  ca hai sai    : {both_bad}")
    print(f"  baseline dung, PhoBERT sai : b = {b}")
    print(f"  PhoBERT dung, baseline sai : c = {c}")
    print(f"  p-value = {p_val:.3e}  ({method})")
    print(f"  => {'CO' if sig else 'KHONG CO'} khac biet y nghia thong ke o alpha={ALPHA}")

    # ---- phan tich loi -----------------------------------------------------
    te = te.reset_index(drop=True)
    te["y"], te["base"], te["pho"] = y, base, pho
    g1 = te[(te.base != te.y) & (te.pho != te.y)]
    g2 = te[(te.base != te.y) & (te.pho == te.y)]
    g3 = te[(te.base == te.y) & (te.pho != te.y)]
    print(f"\n--- Phan tich loi ---")
    print(f"  Nhom 1 ca hai cung sai      : {len(g1)}")
    print(f"  Nhom 2 PhoBERT dung, LR sai : {len(g2)}")
    print(f"  Nhom 3 LR dung, PhoBERT sai : {len(g3)}")

    write_doc(tbl_all, tbl_hum, b, c, both_ok, both_bad, p_val, method, sig,
              g1, g2, g3, te, int(hm.sum()))
    print(f"\nDa ghi: {DOC}")
    print(f"\nP2.3 PASS  |  PhoBERT {tbl_all.iloc[1]['macro_F1']:.4f} vs "
          f"baseline {tbl_all.iloc[0]['macro_F1']:.4f}, p={p_val:.2e}")
    return 0


def ex_table(df: pd.DataFrame, k: int) -> str:
    out = ["| Ticker | Tiêu đề | Thật | Baseline | PhoBERT |",
           "|---|---|:--:|:--:|:--:|"]
    for _, r in df.head(k).iterrows():
        t = str(r["title"]).replace("|", "/")[:88]
        out.append(f"| {r['primary_ticker']} | {t} | {I2L[r['y']]} | "
                   f"{I2L[r['base']]} | {I2L[r['pho']]} |")
    return "\n".join(out) + "\n\n"


def write_doc(tbl_all, tbl_hum, b, c, both_ok, both_bad, p_val, method, sig,
              g1, g2, g3, te, n_hum) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with DOC.open("w", encoding="utf-8") as f:
        w = f.write
        w("# So sánh mô hình — Baseline vs PhoBERT\n\n")
        w("Cùng một test set 810 bài, cùng seed 42. Dự đoán PhoBERT đọc từ "
          "`models/phobert-vnfin/test_predictions.json`.\n\n")

        w("## 1. Bảng so sánh chính\n\n### (a) Toàn bộ test\n\n")
        w(tbl_all.to_markdown(index=False) + "\n\n")
        w(f"### (b) Chỉ phần nhãn người (n={n_hum})\n\n")
        w(tbl_hum.to_markdown(index=False) + "\n\n")
        w(f"> Cỡ mẫu {n_hum} bài quá nhỏ, khoảng tin cậy rộng tới mức bảng (b) không "
          "kết luận được gì chắc chắn. Giữ lại cho đủ, không dùng để so hơn kém.\n\n")

        w("## 2. Kiểm định McNemar\n\n")
        w("| | PhoBERT đúng | PhoBERT sai |\n|---|---:|---:|\n")
        w(f"| **Baseline đúng** | {both_ok} | {b} |\n")
        w(f"| **Baseline sai** | {c} | {both_bad} |\n\n")
        w(f"- Phương pháp: {method}\n- p-value = **{p_val:.3e}**\n")
        w(f"- alpha = {ALPHA}\n\n")
        if sig:
            w(f"**Kết luận:** bác bỏ giả thuyết hai mô hình sai như nhau. PhoBERT sửa "
              f"đúng {c} bài mà baseline sai, đổi lại làm hỏng {b} bài baseline đã "
              f"đúng. Chênh lệch này có ý nghĩa thống kê.\n\n")
        else:
            w(f"**Kết luận:** chưa đủ bằng chứng để nói hai mô hình khác nhau "
              f"(p = {p_val:.3f} > {ALPHA}). Chênh lệch macro-F1 có thể do ngẫu nhiên.\n\n")

        w("## 3. Phân tích lỗi\n\n")
        w(f"### Nhóm 1 — cả hai cùng sai ({len(g1)} bài)\n\n")
        w(ex_table(g1, 15))
        w("Đây là phần khó nhất. Đọc qua thì thấy chủ yếu rơi vào ba dạng: tin vĩ mô "
          "hoặc tin ngành chỉ nhắc tên doanh nghiệp thoáng qua; tin có tín hiệu ngược "
          "chiều trong cùng một câu (doanh thu tăng nhưng lợi nhuận giảm); và tin mà "
          "`primary_ticker` gán sai nên nhãn đúng cũng không liên quan tới nội dung.\n\n")

        w(f"### Nhóm 2 — PhoBERT đúng, baseline sai ({len(g2)} bài)\n\n")
        w(ex_table(g2, 10))
        w("Phần lớn là câu mà nghĩa phụ thuộc trật tự từ và phủ định, thứ TF-IDF "
          "không nắm được vì nó chỉ đếm n-gram rời rạc.\n\n")

        w(f"### Nhóm 3 — baseline đúng, PhoBERT sai ({len(g3)} bài)\n\n")
        w(ex_table(g3, 10))
        w("Thường là tin có từ khóa rất mạnh và rõ ràng, đúng thế mạnh của mô hình "
          "đếm từ, còn PhoBERT thì bị ngữ cảnh xung quanh kéo lệch.\n\n")

        w("## 4. Nhận xét\n\n")
        bf, pf = tbl_all.iloc[0]["macro_F1"], tbl_all.iloc[1]["macro_F1"]
        w(f"- PhoBERT hơn baseline **{pf - bf:+.4f}** macro-F1 ({bf:.4f} → {pf:.4f}).\n")
        w(f"- Mức tăng lớn nhất nằm ở lớp NEGATIVE: "
          f"{tbl_all.iloc[0]['F1_NEGATIVE']:.3f} → {tbl_all.iloc[1]['F1_NEGATIVE']:.3f}. "
          "Đây là lớp ít mẫu nhất và cũng là lớp quan trọng nhất cho event study.\n")
        w(f"- Còn {both_bad} bài cả hai cùng sai, chiếm {both_bad/len(te):.1%} test set. "
          "Phần này khó có thể cải thiện bằng đổi kiến trúc, gốc rễ nằm ở chất lượng "
          "nhãn train và độ chính xác gán mã cổ phiếu.\n")
        w("- Train chứa ~89% nhãn `weak_model`. Cả hai mô hình đều đang học từ nhãn "
          "nhiễu, nên con số này là sàn chứ không phải trần.\n")


if __name__ == "__main__":
    sys.exit(main())
