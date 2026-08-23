"""
eval_labels.py
--------------
Đo mức đồng thuận giữa nhãn máy (labeled_auto.jsonl) và nhãn người (gold_seed.csv).

Điều kiện tiên quyết:
  - data/interim/gold_seed.csv  : cột 'label' đã điền đủ (POSITIVE/NEGATIVE/NEUTRAL)
  - data/interim/labeled_auto.jsonl : kết quả từ prelabel_local.py

Output:
  docs/label_quality.md

Usage:
    python src/eval_labels.py

PASS khi: Cohen's kappa >= 0.6
"""

from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
GOLD_CSV   = ROOT / "data" / "interim" / "gold_seed_v2.csv"
AUTO_JSONL  = ROOT / "data" / "interim" / "labeled_auto.jsonl"
DOCS_OUT   = ROOT / "docs" / "label_quality.md"
LABELS     = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

SEP = "=" * 60

# ---------------------------------------------------------------------------
# 0. Pre-flight checks
# ---------------------------------------------------------------------------
print(SEP)
print("eval_labels.py — Đo chất lượng nhãn máy")
print(SEP)

# --- gold_seed.csv ----------------------------------------------------------
if not GOLD_CSV.exists():
    print(f"\nERROR: Không tìm thấy {GOLD_CSV}")
    print("  Chạy src/prepare_labeling.py trước.")
    sys.exit(1)

gold_df = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")

# Validate label column
filled = gold_df["label"].notna() & (gold_df["label"].astype(str).str.strip() != "")
n_filled = int(filled.sum())
n_total  = len(gold_df)

if n_filled == 0:
    print(f"\nDÙNG LẠI: Cột 'label' trong gold_seed.csv còn trống hoàn toàn ({n_filled}/{n_total}).")
    print()
    print("Hướng dẫn gán nhãn tay:")
    print(f"  1. Mở file: {GOLD_CSV}")
    print("  2. Điền cột 'label' với một trong ba giá trị: POSITIVE, NEUTRAL, NEGATIVE")
    print("     Theo hướng dẫn trong docs/labeling_guide.md")
    print("  3. Lưu file dưới dạng CSV (UTF-8 with BOM) để Excel không mất dấu.")
    print("  4. Chạy lại: python src/eval_labels.py")
    print()
    print("Gợi ý: Bắt đầu với 30–50 bài để kiểm thử nhanh,")
    print("       sau đó hoàn thiện đủ 150 bài.")
    sys.exit(1)

if n_filled < n_total:
    print(f"\nCẢNH BÁO: Chỉ {n_filled}/{n_total} bài đã có nhãn.")
    print("  Tiếp tục với các bài đã điền...")

# Normalise human labels
gold_df["label_human"] = (
    gold_df["label"].astype(str).str.strip().str.upper()
)
valid_mask = gold_df["label_human"].isin(LABELS)
invalid = gold_df[~valid_mask & filled]
if len(invalid) > 0:
    print(f"\nCẢNH BÁO: {len(invalid)} dòng có nhãn không hợp lệ:")
    for _, r in invalid.iterrows():
        print(f"  id={r['id']}  label='{r['label']}'")
    print("  Các dòng này sẽ bị bỏ qua.")

gold_df = gold_df[valid_mask].copy()
print(f"\n  gold_seed: {len(gold_df)} bài có nhãn hợp lệ")

# --- labeled_auto.jsonl ------------------------------------------------------
if not AUTO_JSONL.exists():
    print(f"\nERROR: Không tìm thấy {AUTO_JSONL}")
    print("  Chạy src/prelabel_local.py trước.")
    sys.exit(1)

llm_records: list[dict] = []
with open(AUTO_JSONL, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            try:
                llm_records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

llm_df = pd.DataFrame(llm_records)
llm_df["id"] = llm_df["id"].astype(str)
llm_df["label_llm"] = llm_df["label"].astype(str).str.strip().str.upper()
llm_df = llm_df[llm_df["label_llm"].isin(LABELS)]
print(f"  labeled_auto: {len(llm_df)} bài có nhãn máy")

# ---------------------------------------------------------------------------
# 1. Join
# ---------------------------------------------------------------------------
gold_df["id"] = gold_df["id"].astype(str)
merged = pd.merge(
    gold_df[["id", "title", "primary_ticker", "label_human"]],
    llm_df[["id", "label_llm", "reason", "confidence"]],
    on="id",
    how="inner",
)
print(f"  Matched (inner join): {len(merged)} bài\n")

if len(merged) == 0:
    print("ERROR: Không có bài nào khớp giữa gold_seed và labeled_auto.")
    print("  Kiểm tra cột 'id' trong cả hai file.")
    sys.exit(1)

y_human = merged["label_human"].tolist()
y_llm   = merged["label_llm"].tolist()

# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------
def accuracy(y_true: list, y_pred: list) -> float:
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def cohen_kappa(y_true: list, y_pred: list, labels: list[str]) -> float:
    """Compute Cohen's kappa for multiclass classification."""
    n = len(y_true)
    label_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    # Confusion matrix
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            cm[label_idx[t]][label_idx[p]] += 1

    # Observed agreement
    p_o = sum(cm[i][i] for i in range(k)) / n

    # Expected agreement
    row_sums = [sum(cm[i]) for i in range(k)]
    col_sums = [sum(cm[i][j] for i in range(k)) for j in range(k)]
    p_e = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def precision_recall_f1(y_true: list, y_pred: list, label: str):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


acc   = accuracy(y_human, y_llm)
kappa = cohen_kappa(y_human, y_llm, LABELS)

per_class = {}
for lbl in LABELS:
    prec, rec, f1 = precision_recall_f1(y_human, y_llm, lbl)
    per_class[lbl] = {"precision": prec, "recall": rec, "f1": f1}
macro_f1 = sum(v["f1"] for v in per_class.values()) / len(LABELS)

# ---------------------------------------------------------------------------
# 3. Confusion matrix
# ---------------------------------------------------------------------------
label_idx = {l: i for i, l in enumerate(LABELS)}
k = len(LABELS)
cm = [[0] * k for _ in range(k)]
for t, p in zip(y_human, y_llm):
    if t in label_idx and p in label_idx:
        cm[label_idx[t]][label_idx[p]] += 1

def format_cm_text(cm: list, labels: list) -> str:
    """Return confusion matrix as a readable text block."""
    w = 10
    header = " " * (w + 2) + "  ".join(f"LLM:{l[:3]:>3}" for l in labels)
    lines = [header]
    lines.append(" " * (w + 2) + "-" * (len(header) - w - 2))
    for i, row_label in enumerate(labels):
        row_str = f"Human:{row_label[:3]:>3}  |"
        for j in range(len(labels)):
            mark = " <<" if (i == j) else "   "
            row_str += f"  {cm[i][j]:5d}{mark}"
        lines.append(row_str)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------------------------
print(SEP)
print("KẾT QUẢ ĐÁNH GIÁ")
print(SEP)
print(f"\n  N (matched)   : {len(merged)}")
print(f"  Accuracy      : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Cohen's kappa : {kappa:.4f}")
print(f"  Macro-F1      : {macro_f1:.4f}")

print("\n  Per-class metrics:")
print(f"  {'Label':10s}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}")
print("  " + "-" * 38)
for lbl in LABELS:
    v = per_class[lbl]
    print(f"  {lbl:10s}  {v['precision']:6.3f}  {v['recall']:6.3f}  {v['f1']:6.3f}")

print("\n  Confusion matrix (rows=human, cols=LLM):")
print()
for line in format_cm_text(cm, LABELS).splitlines():
    print("    " + line)
print()

# ---------------------------------------------------------------------------
# 4b. Disagreements
# ---------------------------------------------------------------------------
disagree = merged[merged["label_human"] != merged["label_llm"]].copy()
print(SEP)
print(f"BẤT ĐỒNG: {len(disagree)} / {len(merged)} bài")
print(SEP)

if len(disagree) > 0:
    # Most frequent disagreement type
    disagree["pair"] = disagree.apply(
        lambda r: f"Human={r['label_human']} vs LLM={r['label_llm']}", axis=1
    )
    pair_counts = Counter(disagree["pair"])
    print("\nNhóm bất đồng phổ biến nhất:")
    for pair, cnt in pair_counts.most_common():
        pct = cnt / len(merged) * 100
        print(f"  {pair:<40s} {cnt:3d} bài  ({pct:.1f}%)")

    print(f"\n--- Chi tiết {len(disagree)} bài bất đồng ---\n")
    for idx, (_, row) in enumerate(disagree.iterrows(), 1):
        ticker = str(row.get("primary_ticker", "?"))
        title  = str(row.get("title", ""))[:100]
        reason = str(row.get("reason", ""))[:120]
        print(f"[{idx:3d}] Ticker: {ticker}")
        print(f"       Title : {title}")
        print(f"       Human : {row['label_human']}")
        print(f"       LLM   : {row['label_llm']}  (conf={row.get('confidence',0):.2f})")
        print(f"       Reason: {reason}")
        print()

# ---------------------------------------------------------------------------
# 5. PASS / FAIL
# ---------------------------------------------------------------------------
print(SEP)
PASS = True
KAPPA_THRESHOLD = 0.6

if kappa >= KAPPA_THRESHOLD:
    print(f"PASS: Cohen's kappa = {kappa:.4f} >= {KAPPA_THRESHOLD}")
else:
    print(f"FAIL: Cohen's kappa = {kappa:.4f} < {KAPPA_THRESHOLD}")
    PASS = False

    # Analyse biggest disagreement group
    if len(disagree) > 0:
        top_pair, top_cnt = pair_counts.most_common(1)[0]
        print(f"\n  Nhóm bất đồng lớn nhất: {top_pair} ({top_cnt} bài)")
        # Parse the pair
        h_lbl = top_pair.split("Human=")[1].split(" vs")[0]
        l_lbl = top_pair.split("LLM=")[1]
        print()
        print("  Phân tích và đề xuất sửa guideline:")
        suggestions = {
            ("POSITIVE", "NEUTRAL"): (
                "LLM NEUTRAL khi người gán POSITIVE. "
                "Nguyên nhân thường: tin có dấu hiệu POSITIVE nhưng thiếu con số cụ thể. "
                "Đề xuất: Thêm ví dụ 'POSITIVE không cần số tuyệt đối' vào Section 7 guideline. "
                "Xem xét hạ ngưỡng cụ thể hóa (specificity threshold) cho mảng ký hợp đồng/IPO."
            ),
            ("NEUTRAL", "POSITIVE"): (
                "LLM POSITIVE khi người gán NEUTRAL. "
                "Nguyên nhân thường: LLM bị ảnh hưởng bởi từ ngữ tích cực trong PR/giải thưởng. "
                "Đề xuất: Bổ sung thêm ví dụ PR thuần vào Case 7, nhấn mạnh 'không có số liệu tài chính → NEUTRAL'."
            ),
            ("NEGATIVE", "NEUTRAL"): (
                "LLM NEUTRAL khi người gán NEGATIVE. "
                "Nguyên nhân thường: tin cảnh báo nhẹ, cắt margin gián tiếp. "
                "Đề xuất: Bổ sung rule rõ hơn cho Case 5 — cụ thể hóa 'cảnh báo gián tiếp' cũng là NEGATIVE."
            ),
            ("NEUTRAL", "NEGATIVE"): (
                "LLM NEGATIVE khi người gán NEUTRAL. "
                "Nguyên nhân thường: LLM quá nhạy với từ ngữ tiêu cực trong tin vĩ mô. "
                "Đề xuất: Nhấn mạnh Case 9 (tin vĩ mô) và quy tắc 'phân vân → NEUTRAL' trong system prompt."
            ),
            ("POSITIVE", "NEGATIVE"): (
                "Bất đồng cực đại. "
                "Nguyên nhân thường: bài chứa cả tín hiệu tích cực lẫn tiêu cực. "
                "Đề xuất: Bổ sung quy tắc 'tín hiệu mâu thuẫn → NEUTRAL' vào Section 6."
            ),
            ("NEGATIVE", "POSITIVE"): (
                "Bất đồng cực đại. "
                "Nguyên nhân thường: LLM bỏ qua tín hiệu tiêu cực ẩn. "
                "Đề xuất: Thêm ví dụ lãnh đạo bán cổ phiếu cùng tin tốt vào Case 2."
            ),
        }
        key = (h_lbl, l_lbl)
        suggestion = suggestions.get(
            key,
            f"Xem xét lại {top_cnt} case {top_pair} và bổ sung ví dụ vào guideline."
        )
        for line in suggestion.split(". "):
            if line.strip():
                print(f"    • {line.strip()}.")
print()

# ---------------------------------------------------------------------------
# 6. Write docs/label_quality.md
# ---------------------------------------------------------------------------
DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)

from datetime import datetime

with open(DOCS_OUT, "w", encoding="utf-8") as fh:
    def w(s=""):
        fh.write(s + "\n")

    w("# Báo cáo Chất lượng Nhãn LLM")
    w()
    w(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    w(f"**Gold set:** `data/interim/gold_seed.csv` ({len(merged)} bài khớp)  ")
    w(f"**Nhãn máy:** `data/interim/labeled_auto.jsonl`")
    w()
    w("---")
    w()

    # Summary metrics
    w("## Chỉ số tổng hợp")
    w()
    w("| Chỉ số | Giá trị |")
    w("|--------|--------:|")
    w(f"| N (matched) | {len(merged)} |")
    w(f"| Accuracy | {acc:.4f} ({acc*100:.1f}%) |")
    w(f"| **Cohen's kappa** | **{kappa:.4f}** |")
    w(f"| Macro-F1 | {macro_f1:.4f} |")
    w()

    # Per-class
    w("## Chỉ số theo nhãn")
    w()
    w("| Nhãn | Precision | Recall | F1 |")
    w("|------|----------:|-------:|---:|")
    for lbl in LABELS:
        v = per_class[lbl]
        w(f"| {lbl} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    w()

    # Confusion matrix
    w("## Confusion Matrix")
    w()
    w("> Hàng = nhãn người (Human), Cột = nhãn máy")
    w()
    w("| Human \\ LLM | " + " | ".join(f"**{l}**" for l in LABELS) + " |")
    w("|-------------|" + "|".join(["------:"] * k) + "|")
    for i, row_label in enumerate(LABELS):
        row_cells = " | ".join(str(cm[i][j]) for j in range(k))
        w(f"| **{row_label}** | {row_cells} |")
    w()

    # Label distribution comparison
    human_cnt = Counter(y_human)
    llm_cnt   = Counter(y_llm)
    w("## Phân bố nhãn")
    w()
    w("| Nhãn | Người (%) | LLM (%) |")
    w("|------|----------:|--------:|")
    for lbl in LABELS:
        h_pct = human_cnt.get(lbl, 0) / len(y_human) * 100
        l_pct = llm_cnt.get(lbl, 0)   / len(y_llm)   * 100
        w(f"| {lbl} | {h_pct:.1f}% | {l_pct:.1f}% |")
    w()

    # Disagreements
    w("## Bất đồng chi tiết")
    w()
    w(f"Tổng số bất đồng: **{len(disagree)}** / {len(merged)} bài ({len(disagree)/len(merged)*100:.1f}%)")
    w()
    if len(disagree) > 0:
        w("### Nhóm bất đồng")
        w()
        w("| Human → LLM | Số bài | Tỷ lệ |")
        w("|-------------|-------:|------:|")
        for pair, cnt in pair_counts.most_common():
            w(f"| {pair} | {cnt} | {cnt/len(merged)*100:.1f}% |")
        w()

        w("### Chi tiết từng case bất đồng")
        w()
        for idx, (_, row) in enumerate(disagree.iterrows(), 1):
            ticker = str(row.get("primary_ticker", "?"))
            title  = str(row.get("title", ""))
            reason = str(row.get("reason", ""))
            conf   = float(row.get("confidence", 0))
            w(f"**[{idx}]** `{ticker}` — {title}  ")
            w(f"- Người: `{row['label_human']}` | LLM: `{row['label_llm']}` (conf={conf:.2f})  ")
            w(f"- LLM reason: *{reason}*")
            w()

    # Pass/Fail
    status = "✅ PASS" if PASS else "❌ FAIL"
    w("## Kết quả kiểm tra")
    w()
    w(f"**{status}**")
    w()
    w(f"- Cohen's kappa = **{kappa:.4f}** (ngưỡng >= {KAPPA_THRESHOLD})")
    w()

    if not PASS and len(disagree) > 0:
        top_pair, top_cnt = pair_counts.most_common(1)[0]
        h_lbl = top_pair.split("Human=")[1].split(" vs")[0]
        l_lbl = top_pair.split("LLM=")[1]
        w("## Đề xuất cải thiện Guideline")
        w()
        w(f"> Nhóm bất đồng lớn nhất: **{top_pair}** ({top_cnt} bài)")
        w()
        suggestion = suggestions.get(
            (h_lbl, l_lbl),
            f"Xem xét lại {top_cnt} case {top_pair} và bổ sung ví dụ vào guideline."
        )
        for line in suggestion.split(". "):
            if line.strip():
                w(f"- {line.strip()}.")
        w()
        w("> **Lưu ý:** Không tự sửa guideline — cần review thủ công trước khi cập nhật.")

print(f"Đã ghi: {DOCS_OUT}")
print(SEP)

if not PASS:
    sys.exit(1)