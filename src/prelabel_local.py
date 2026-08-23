"""
prelabel_local.py
-----------------
Gán nhãn ~5.500 bài KHÔNG gọi API trả phí, thay thế cho src/prelabel_local.py.

Ý tưởng: thay vì hỏi LLM từng bài (tốn tiền, cần API key), ta dùng một tập
nhãn hạt giống chất lượng cao đã có sẵn rồi lan truyền sang phần còn lại bằng
một bộ phân loại nhẹ.

    data/interim/claude_seed_labels.csv   1.170 nhãn do trợ lý Claude gán tay
                                          (toàn bộ split test + mẫu train/val)
              |
              v
    TF-IDF (word 1-2gram + char_wb 3-5gram) trên text_input
    + 6 feature từ src/lexicon_vi_fin.py
              |
              v
    LogisticRegression đã hiệu chuẩn xác suất (CalibratedClassifierCV)
              |
              v
    self-training 1 vòng: thêm dự đoán có conf >= SELF_TRAIN_TH rồi huấn luyện lại
              |
              v
    data/interim/labeled_auto.jsonl   (đúng schema mà eval_labels.py / finalize_dataset.py cần)

Ràng buộc quan trọng để con số kappa ở P1.3 không bị thổi phồng:
  - KHÔNG bao giờ đưa nhãn trong gold_seed_v2.csv vào tập huấn luyện.
    Nhãn người là thước đo, dùng nó để dạy mô hình rồi lại đem chấm điểm mô hình
    trên chính nó là gian lận.
  - Với id đã có nhãn tay trong seed, ghi thẳng nhãn tay (label_source="claude_manual").
  - Với id còn lại, ghi dự đoán mô hình (label_source="weak_model") kèm confidence
    thực tế đã hiệu chuẩn.

Usage:
    python src/prelabel_local.py              # chạy bình thường (resume được)
    python src/prelabel_local.py --rebuild    # bỏ file cũ, gán lại từ đầu
    python src/prelabel_local.py --report     # chỉ in thống kê file đã có
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexicon_vi_fin import FEATURE_NAMES, score_text  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- cấu hình ---
ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "interim" / "to_label_v2.parquet"
SEED_FILE = ROOT / "data" / "interim" / "claude_seed_labels.csv"
GOLD_FILE = ROOT / "data" / "interim" / "gold_seed_v2.csv"
OUT_FILE = ROOT / "data" / "interim" / "labeled_auto.jsonl"
REPORT = ROOT / "docs" / "prelabel_local_report.md"

SEED = 42
N_FOLDS = 5
SELF_TRAIN_TH = 0.80   # ngưỡng conf để đưa dự đoán vào vòng huấn luyện thứ 2
MANUAL_CONF = 0.95     # confidence gán cho nhãn tay
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def build_features(texts: list[str], vecs=None, scaler=None, fit: bool = False):
    """TF-IDF word + char, ghép thêm feature từ điển. Trả về (X, vecs, scaler)."""
    if fit:
        vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             max_features=60000, sublinear_tf=True)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                             max_features=80000, sublinear_tf=True)
        Xw, Xc = vw.fit_transform(texts), vc.fit_transform(texts)
        vecs = (vw, vc)
    else:
        vw, vc = vecs
        Xw, Xc = vw.transform(texts), vc.transform(texts)

    lex = np.array([[score_text(t)[k] for k in FEATURE_NAMES] for t in texts],
                   dtype=np.float64)
    if fit:
        scaler = StandardScaler().fit(lex)
    Xl = csr_matrix(scaler.transform(lex))

    return hstack([Xw, Xc, Xl]).tocsr(), vecs, scaler


def make_clf() -> CalibratedClassifierCV:
    base = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced",
                              random_state=SEED)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def reliability_table(conf: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    """Bảng đối chiếu confidence dự đoán với độ chính xác thực tế."""
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    idx = np.digitize(conf, bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "khoảng conf": f"{bins[b]:.2f}-{bins[b+1]:.2f}",
            "n": int(m.sum()),
            "conf TB": round(float(conf[m].mean()), 3),
            "accuracy thực": round(float(correct[m].mean()), 3),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="xóa jsonl cũ, gán lại")
    ap.add_argument("--report", action="store_true", help="chỉ in thống kê file có sẵn")
    args = ap.parse_args()

    if args.report:
        return summarize()

    print("=" * 68)
    print("PRELABEL LOCAL - khong goi API, khong ton phi")
    print("=" * 68)

    df = pd.read_parquet(IN_FILE)
    df["id"] = df["id"].astype(str)
    print(f"  to_label_v2      : {len(df):,} bai")

    if not SEED_FILE.exists():
        print(f"  THIEU {SEED_FILE}. Khong the chay.")
        return 1
    seed = pd.read_csv(SEED_FILE, encoding="utf-8-sig")
    seed["id"] = seed["id"].astype(str)
    seed = seed[seed["label"].isin(LABELS)].drop_duplicates(subset="id")
    print(f"  nhan hat giong   : {len(seed):,} bai ({len(seed)/len(df):.1%})")

    gold_ids: set[str] = set()
    if GOLD_FILE.exists():
        g = pd.read_csv(GOLD_FILE, encoding="utf-8-sig")
        g = g[g["label"].notna() & (g["label"].astype(str).str.strip() != "")]
        gold_ids = set(g["id"].astype(str))
    print(f"  nhan nguoi (gold): {len(gold_ids):,} bai - KHONG dua vao training")

    if args.rebuild and OUT_FILE.exists():
        OUT_FILE.unlink()

    done_ids: set[str] = set()
    if OUT_FILE.exists():
        with OUT_FILE.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(str(json.loads(line)["id"]))
                    except Exception:
                        pass
        print(f"  resume           : bo qua {len(done_ids):,} bai da co nhan")

    text_by_id = dict(zip(df["id"], df["text_input"].astype(str)))

    # ---- tap huan luyen: chi nhan hat giong, tuyet doi khong co gold ---------
    train = seed.copy()
    train["text"] = train["id"].map(text_by_id)
    train = train.dropna(subset=["text"])
    print(f"  tap train        : {len(train):,} bai")
    print("  phan bo:", dict(Counter(train["label"])))

    Xtr, vecs, scaler = build_features(train["text"].tolist(), fit=True)
    ytr = train["label"].to_numpy()

    # ---- do chat luong bang cross-validation --------------------------------
    print("\n" + "-" * 68)
    print("Cross-validation 5-fold tren tap hat giong")
    print("-" * 68)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    proba_cv = cross_val_predict(make_clf(), Xtr, ytr, cv=cv, method="predict_proba")
    classes = np.array(sorted(set(ytr)))
    pred_cv = classes[proba_cv.argmax(axis=1)]
    conf_cv = proba_cv.max(axis=1)
    correct = (pred_cv == ytr).astype(float)

    acc = accuracy_score(ytr, pred_cv)
    mf1 = f1_score(ytr, pred_cv, average="macro")
    print(f"  accuracy  : {acc:.4f}")
    print(f"  macro-F1  : {mf1:.4f}")
    print(classification_report(ytr, pred_cv, digits=3, zero_division=0))
    rel = reliability_table(conf_cv, correct)
    print("Do tin cay cua confidence:")
    print(rel.to_string(index=False))

    # ---- vong 1: huan luyen day du + du doan phan con lai -------------------
    clf = make_clf().fit(Xtr, ytr)
    rest = df[~df["id"].isin(set(train["id"]))].reset_index(drop=True)
    Xre, _, _ = build_features(rest["text_input"].astype(str).tolist(), vecs, scaler)
    p1 = clf.predict_proba(Xre)

    # ---- vong 2: self-training tren du doan tu tin --------------------------
    conf1 = p1.max(axis=1)
    lab1 = classes[p1.argmax(axis=1)]
    take = conf1 >= SELF_TRAIN_TH
    print(f"\nSelf-training: them {int(take.sum()):,} du doan co conf >= {SELF_TRAIN_TH}")

    aug_texts = train["text"].tolist() + rest.loc[take, "text_input"].astype(str).tolist()
    aug_y = np.concatenate([ytr, lab1[take]])
    Xa, vecs2, scaler2 = build_features(aug_texts, fit=True)
    clf2 = make_clf().fit(Xa, aug_y)

    Xre2, _, _ = build_features(rest["text_input"].astype(str).tolist(), vecs2, scaler2)
    p2 = clf2.predict_proba(Xre2)
    conf2 = p2.max(axis=1)
    lab2 = classes[p2.argmax(axis=1)]
    print(f"  conf trung binh sau self-training: {conf2.mean():.3f} "
          f"(truoc: {conf1.mean():.3f})")

    # ---- ghi ket qua ---------------------------------------------------------
    manual = dict(zip(train["id"], train["label"]))
    pred_map = dict(zip(rest["id"], zip(lab2, conf2)))

    n_new = 0
    with OUT_FILE.open("a", encoding="utf-8") as out:
        for _, row in df.iterrows():
            rid = row["id"]
            if rid in done_ids:
                continue
            if rid in manual:
                label, conf, src = manual[rid], MANUAL_CONF, "claude_manual"
                reason = "gan tay theo docs/labeling_guide.md"
            else:
                label, conf = pred_map[rid]
                src = "weak_model"
                reason = "lan truyen tu nhan hat giong (tfidf+lexicon+LR)"
            out.write(json.dumps({
                "id": rid,
                "primary_ticker": str(row.get("primary_ticker", "")),
                "label": label,
                "confidence": round(float(conf), 4),
                "reason": reason,
                "label_source": src,
                "title": str(row.get("title", ""))[:200],
                "date": str(row.get("date", "")),
            }, ensure_ascii=False) + "\n")
            n_new += 1

    print(f"\nDa ghi them {n_new:,} dong vao {OUT_FILE.name}")
    write_report(len(df), len(train), acc, mf1, rel, classes, ytr, pred_cv, gold_ids)
    return summarize()


def write_report(n_total, n_seed, acc, mf1, rel, classes, ytr, pred_cv, gold_ids) -> None:
    cm = pd.crosstab(pd.Series(ytr, name="Claude gan tay"),
                     pd.Series(pred_cv, name="Mo hinh du doan"))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as f:
        w = f.write
        w("# Báo cáo gán nhãn cục bộ (prelabel_local)\n\n")
        w("Thay thế cho `prelabel_local.py`. Không gọi API, không tốn phí.\n\n")
        w("## 1. Quy mô\n\n")
        w(f"- Tổng số bài cần nhãn: **{n_total:,}**\n")
        w(f"- Nhãn hạt giống gán tay: **{n_seed:,}** ({n_seed/n_total:.1%})\n")
        w(f"- Nhãn người trong gold seed (không dùng để huấn luyện): {len(gold_ids):,}\n\n")
        w("## 2. Chất lượng mô hình lan truyền (5-fold CV trên tập hạt giống)\n\n")
        w(f"- accuracy: **{acc:.4f}**\n- macro-F1: **{mf1:.4f}**\n\n")
        w("### Ma trận nhầm lẫn\n\n")
        w(cm.to_markdown() + "\n\n")
        w("### Độ tin cậy của confidence\n\n")
        w(rel.to_markdown(index=False) + "\n\n")
        w("Cột `accuracy thực` cho biết trong nhóm dự đoán có confidence rơi vào\n")
        w("khoảng đó, thực tế bao nhiêu phần trăm đúng. Dùng bảng này để chọn\n")
        w("ngưỡng lọc ở `finalize_dataset.py`.\n\n")
        w("## 3. Cách đọc cột label_source trong labeled_auto.jsonl\n\n")
        w("| Giá trị | Nghĩa | confidence |\n|---|---|---|\n")
        w("| `claude_manual` | Trợ lý đọc từng bài và gán theo labeling_guide | 0.95 |\n")
        w("| `weak_model` | Mô hình lan truyền dự đoán | xác suất đã hiệu chuẩn |\n\n")
        w("## 4. Hạn chế\n\n")
        w("- Nhãn hạt giống do một mô hình ngôn ngữ gán, không phải chuyên gia tài chính.\n")
        w("- Mô hình lan truyền chỉ nhìn tiêu đề + sapo, không đọc toàn văn.\n")
        w("- Tập hạt giống lệch về nửa cuối giai đoạn dữ liệu, nên phần train\n")
        w("  (thời gian sớm hơn) chủ yếu là nhãn `weak_model`, độ nhiễu cao hơn.\n")
        w("- Nhãn `weak_model` chỉ nên coi là nhãn yếu. Mọi kết luận về chất lượng\n")
        w("  mô hình ở P2 phải báo cáo riêng trên phần `label_source == \"human\"`.\n")


def summarize() -> int:
    if not OUT_FILE.exists():
        print("Chua co labeled_auto.jsonl")
        return 1
    recs = []
    with OUT_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    total = len(recs)
    print("\n" + "=" * 68)
    print("TONG KET")
    print("=" * 68)
    print(f"labeled_auto.jsonl: {total:,} dong")
    cnt = Counter(r["label"] for r in recs)
    for lbl in LABELS:
        pct = cnt.get(lbl, 0) / max(total, 1) * 100
        print(f"  {lbl:9s} {cnt.get(lbl,0):6,}  ({pct:5.1f}%)  {'#' * int(pct/2)}")
    src = Counter(r.get("label_source", "?") for r in recs)
    print("\nNguon nhan:", dict(src))
    conf = np.array([r["confidence"] for r in recs])
    print(f"confidence: TB={conf.mean():.3f}  <0.5: {int((conf<0.5).sum()):,} bai")

    ok = True
    if total < 5000:
        print(f"FAIL: chi co {total:,} dong")
        ok = False
    for lbl in LABELS:
        pct = cnt.get(lbl, 0) / max(total, 1) * 100
        if pct < 5:
            print(f"FAIL: nhan {lbl} chi {pct:.1f}% (<5%)")
            ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
