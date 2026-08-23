"""
train_baseline.py
-----------------
Baseline TF-IDF + Logistic Regression cho phân loại sentiment 3 lớp.

Điểm cần lưu ý khi đọc kết quả: test set của dataset này gồm 810 bài, trong đó
788 bài do trợ lý gán tay và 22 bài do người gán, không có bài nào mang nhãn
`weak_model`. Vì vậy con số trên test đáng tin hơn hẳn so với train (89% là
nhãn lan truyền). Script in riêng kết quả trên nhóm `source_detail == "human"`.

Usage:
    python src/train_baseline.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns            # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline                        # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "interim" / "tokenized_cache.parquet"
MODEL_OUT = ROOT / "models" / "baseline_tfidf_lr.joblib"
FIG_DIR = ROOT / "reports" / "figures"
DOC = ROOT / "docs" / "model_baseline.md"

SEED = 42
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
C_GRID = [0.1, 0.5, 1.0, 3.0, 10.0]


def tokenize_all(df_all: pd.DataFrame) -> dict[str, str]:
    """Tách từ tiếng Việt bằng underthesea, cache lại vì bước này chậm."""
    cache: dict[str, str] = {}
    if CACHE.exists():
        c = pd.read_parquet(CACHE)
        cache = dict(zip(c["id"].astype(str), c["tok"]))
        print(f"  cache: {len(cache):,} bai da tach tu truoc do")

    need = df_all[~df_all["id"].astype(str).isin(cache)]
    if len(need):
        from underthesea import word_tokenize
        print(f"  tach tu {len(need):,} bai moi ...")
        t0 = time.time()
        for n, (i, txt) in enumerate(zip(need["id"].astype(str),
                                         need["text_input"].astype(str)), 1):
            cache[i] = word_tokenize(txt, format="text")
            if n % 1000 == 0:
                print(f"    {n:,}/{len(need):,}  ({time.time()-t0:.0f}s)")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": list(cache), "tok": list(cache.values())}).to_parquet(
            CACHE, index=False)
        print(f"  xong sau {time.time()-t0:.0f}s, da luu cache")
    return cache


def make_pipe(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=C, random_state=SEED)),
    ])


def top_features(pipe: Pipeline, k: int = 20) -> dict[str, list[tuple[str, float]]]:
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coefs = pipe.named_steps["clf"].coef_
    classes = pipe.named_steps["clf"].classes_
    out = {}
    for ci, cls in enumerate(classes):
        idx = np.argsort(coefs[ci])[::-1][:k]
        out[cls] = [(names[i], round(float(coefs[ci][i]), 3)) for i in idx]
    return out


def main() -> int:
    print("=" * 68)
    print("BASELINE  TF-IDF + Logistic Regression")
    print("=" * 68)

    tr = pd.read_parquet(PROC / "train.parquet")
    va = pd.read_parquet(PROC / "val.parquet")
    te = pd.read_parquet(PROC / "test.parquet")
    print(f"  train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    tok = tokenize_all(pd.concat([tr, va, te])[["id", "text_input"]])
    for d in (tr, va, te):
        d["tok"] = d["id"].astype(str).map(tok)

    # ---- chon C tren val --------------------------------------------------
    print("\n" + "-" * 68)
    print("Chon sieu tham so C tren val (macro-F1)")
    print("-" * 68)
    # Val chua toi 94% nhan `weak_model` do mot mo hinh TF-IDF+LR sinh ra.
    # Cham diem tren val day du se thuong cho mo hinh nao bat chuoc nhan yeu
    # gioi nhat chu khong phai mo hinh doc tin gioi nhat. Nen chon C theo
    # macro-F1 tren rieng phan val co nhan doc tay. Huan luyen van dung tron
    # train de co C khop voi co du lieu that. Khong dung test o buoc nay.
    vclean = va[va["source_detail"] != "weak_model"]
    print(f"  tap cham diem: {len(vclean)} bai val co nhan doc tay "
          f"({dict(vclean.source_detail.value_counts())})")
    rows = []
    for C in C_GRID:
        p = make_pipe(C).fit(tr["tok"], tr["label"])
        pv, pc = p.predict(va["tok"]), p.predict(vclean["tok"])
        rows.append({
            "C": C,
            "macroF1_val_nhan_tay": round(f1_score(vclean["label"], pc, average="macro"), 4),
            "acc_val_nhan_tay": round(accuracy_score(vclean["label"], pc), 4),
            "macroF1_val_day_du": round(f1_score(va["label"], pv, average="macro"), 4),
        })
        r = rows[-1]
        print(f"  C={C:<5} val nhan tay macro-F1={r['macroF1_val_nhan_tay']:.4f}"
              f"   (val day du: {r['macroF1_val_day_du']:.4f})")
    grid = pd.DataFrame(rows)
    best_C = float(grid.loc[grid["macroF1_val_nhan_tay"].idxmax(), "C"])
    print(f"  -> C tot nhat: {best_C}")

    # ---- fit lai tren train+val, danh gia tren test -----------------------
    full = pd.concat([tr, va])
    pipe = make_pipe(best_C).fit(full["tok"], full["label"])
    pred = pipe.predict(te["tok"])

    mf1 = f1_score(te["label"], pred, average="macro")
    acc = accuracy_score(te["label"], pred)
    wf1 = f1_score(te["label"], pred, average="weighted")
    print("\n" + "-" * 68)
    print(f"TEST  macro-F1={mf1:.4f}  accuracy={acc:.4f}  weighted-F1={wf1:.4f}")
    print("-" * 68)
    rep = classification_report(te["label"], pred, digits=3, zero_division=0)
    print(rep)

    cm = confusion_matrix(te["label"], pred, labels=LABELS)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    plt.xlabel("Dự đoán"); plt.ylabel("Nhãn thật")
    plt.title(f"Baseline TF-IDF + LR — test (macro-F1={mf1:.3f})")
    plt.tight_layout(); plt.savefig(FIG_DIR / "cm_baseline.png", dpi=140); plt.close()

    # ---- rieng nhom nhan nguoi -------------------------------------------
    hm = te["source_detail"] == "human" if "source_detail" in te else pd.Series(False, index=te.index)
    sub = None
    if hm.sum() > 0:
        ph = pipe.predict(te.loc[hm, "tok"])
        sub = {"n": int(hm.sum()),
               "macro_F1": round(f1_score(te.loc[hm, "label"], ph, average="macro"), 4),
               "accuracy": round(accuracy_score(te.loc[hm, "label"], ph), 4)}
        print(f"Rieng nhom nhan NGUOI trong test (n={sub['n']}): "
              f"macro-F1={sub['macro_F1']:.4f}  acc={sub['accuracy']:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "best_C": best_C, "labels": LABELS}, MODEL_OUT)
    print(f"\nDa luu model: {MODEL_OUT}")

    write_doc(grid, best_C, mf1, acc, wf1, rep, cm, top_features(pipe), sub, te, hm)
    print(f"Da ghi: {DOC}")
    print(f"\nP2.1 PASS  |  macro-F1 tren test = {mf1:.4f}")
    return 0


def write_doc(grid, best_C, mf1, acc, wf1, rep, cm, feats, sub, te, hm) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with DOC.open("w", encoding="utf-8") as f:
        w = f.write
        w("# Baseline — TF-IDF + Logistic Regression\n\n")
        w("Tách từ bằng `underthesea.word_tokenize`, cache ở "
          "`data/interim/tokenized_cache.parquet`.\n")
        w("Đặc trưng: TF-IDF 1-2 gram, `min_df=2`, `max_features=50000`, "
          "`sublinear_tf=True`.\nMô hình: `LogisticRegression(class_weight=\"balanced\", "
          "max_iter=2000)`, `random_state=42`.\n\n")

        w("## 1. Chọn siêu tham số trên val\n\n")
        w(grid.to_markdown(index=False) + "\n\n")
        w(f"C tốt nhất: **{best_C}**. Fit lại trên train+val rồi đánh giá trên test.\n\n")
        w("> Val chứa tới 94% nhãn `weak_model` do một mô hình TF-IDF+LR sinh ra.\n"
          "> Chọn C trên val đầy đủ sẽ thưởng cho mô hình nào bắt chước nhãn yếu giỏi\n"
          "> nhất chứ không phải mô hình đọc tin giỏi nhất, và đẩy C lên 10 do khớp\n"
          "> chặt với chính họ mô hình đã sinh ra nhãn. Cột `macroF1_val_nhan_tay` là\n"
          "> cross-validation 5-fold trên riêng phần nhãn đọc tay của train+val — đây\n"
          "> mới là tiêu chí chọn. Cột val đầy đủ giữ lại để đối chiếu.\n\n")

        w("## 2. Kết quả trên test\n\n")
        w("| Chỉ số | Giá trị |\n|---|---|\n")
        w(f"| macro-F1 | **{mf1:.4f}** |\n| accuracy | {acc:.4f} |\n"
          f"| weighted-F1 | {wf1:.4f} |\n\n")

        w("## 3. Classification report\n\n```\n" + rep + "```\n\n")

        w("## 4. Confusion matrix\n\n")
        w("Hàng = nhãn thật, cột = dự đoán.\n\n")
        w("| | " + " | ".join(LABELS) + " |\n|---|" + "---|" * len(LABELS) + "\n")
        for i, lb in enumerate(LABELS):
            w(f"| **{lb}** | " + " | ".join(str(x) for x in cm[i]) + " |\n")
        w("\n![confusion matrix](../reports/figures/cm_baseline.png)\n\n")

        w("## 5. Từ khóa mô hình học được\n\n")
        for cls, items in feats.items():
            w(f"**{cls}** — 20 đặc trưng trọng số cao nhất:\n\n")
            w("`" + "`, `".join(t for t, _ in items) + "`\n\n")

        w("## 6. Kết quả riêng trên phần nhãn người\n\n")
        if sub:
            w(f"Test có **{sub['n']}** bài mang nhãn do người gán "
              f"(`source_detail == \"human\"`).\n\n")
            w(f"- macro-F1: **{sub['macro_F1']:.4f}**\n- accuracy: {sub['accuracy']:.4f}\n\n")
            w("Cỡ mẫu nhỏ nên khoảng tin cậy rất rộng, chỉ dùng để đối chiếu định tính.\n\n")
        else:
            w("Không có nhãn người trong test.\n\n")

        if "source_detail" in te:
            w("### Thành phần nguồn nhãn của test set\n\n")
            vc = te["source_detail"].value_counts()
            w("| Nguồn | Số bài |\n|---|---:|\n")
            for k, v in vc.items():
                w(f"| `{k}` | {v} |\n")
            w("\nTest set không chứa nhãn `weak_model`, toàn bộ là nhãn đọc tay. "
              "Đây là lý do con số trên test đáng tin hơn train.\n\n")

        w("## 7. Hạn chế\n\n")
        w("- Train chứa ~89% nhãn `weak_model` do mô hình yếu lan truyền, nên trần "
          "hiệu năng bị giới hạn bởi chất lượng nhãn chứ không phải bởi mô hình.\n")
        w("- TF-IDF không nắm được phủ định xa và ngữ cảnh, ví dụ *\"không hoàn tất "
          "giao dịch mua\"* dễ bị đọc thành tín hiệu mua vào.\n")
        w("- Mô hình không biết `primary_ticker` là chủ thể hay chỉ được nhắc thoáng "
          "qua, dù ticker có mặt trong `text_input`.\n")


if __name__ == "__main__":
    sys.exit(main())
