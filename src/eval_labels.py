from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
GOLD_CSV   = ROOT / "data" / "interim" / "gold_seed_v2.csv"
AUTO_JSONL  = ROOT / "data" / "interim" / "labeled_auto.jsonl"
DOCS_OUT   = ROOT / "docs" / "label_quality.md"
LABELS     = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

SEP = "=" * 60

print(SEP)
print("eval_labels.py — Label quality evaluation")
print(SEP)

if not GOLD_CSV.exists():
    print(f"\nERROR: {GOLD_CSV} not found")
    print("  Run src/prepare_labeling.py first.")
    sys.exit(1)

gold_df = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")

filled = gold_df["label"].notna() & (gold_df["label"].astype(str).str.strip() != "")
n_filled = int(filled.sum())
n_total  = len(gold_df)

if n_filled == 0:
    print(f"\nNOTHING TO DO: 'label' column in gold_seed.csv is completely empty ({n_filled}/{n_total}).")
    print()
    print("Manual labeling instructions:")
    print(f"  1. Open file: {GOLD_CSV}")
    print("  2. Fill the 'label' column with one of: POSITIVE, NEUTRAL, NEGATIVE")
    print("     Following the guide in docs/labeling_guide.md")
    print("  3. Save as CSV (UTF-8 with BOM) so Excel handles accents correctly.")
    print("  4. Re-run: python src/eval_labels.py")
    print()
    print("Tip: Start with 30-50 articles for a quick check,")
    print("     then complete all 150.")
    sys.exit(1)

if n_filled < n_total:
    print(f"\nWARNING: Only {n_filled}/{n_total} articles have labels.")
    print("  Continuing with labeled articles...")

gold_df["label_human"] = (
    gold_df["label"].astype(str).str.strip().str.upper()
)
valid_mask = gold_df["label_human"].isin(LABELS)
invalid = gold_df[~valid_mask & filled]
if len(invalid) > 0:
    print(f"\nWARNING: {len(invalid)} rows have invalid labels:")
    for _, r in invalid.iterrows():
        print(f"  id={r['id']}  label='{r['label']}'")
    print("  These rows will be ignored.")

gold_df = gold_df[valid_mask].copy()
print(f"\n  gold_seed: {len(gold_df)} articles with valid labels")

if not AUTO_JSONL.exists():
    print(f"\nERROR: {AUTO_JSONL} not found")
    print("  Run src/prelabel_local.py first.")
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
print(f"  labeled_auto: {len(llm_df)} articles with machine labels")

gold_df["id"] = gold_df["id"].astype(str)
merged = pd.merge(
    gold_df[["id", "title", "primary_ticker", "label_human"]],
    llm_df[["id", "label_llm", "reason", "confidence"]],
    on="id",
    how="inner",
)
print(f"  Matched (inner join): {len(merged)} articles\n")

if len(merged) == 0:
    print("ERROR: No articles matched between gold_seed and labeled_auto.")
    print("  Check the 'id' column in both files.")
    sys.exit(1)

y_human = merged["label_human"].tolist()
y_llm   = merged["label_llm"].tolist()


def accuracy(y_true: list, y_pred: list) -> float:
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def cohen_kappa(y_true: list, y_pred: list, labels: list[str]) -> float:
    n = len(y_true)
    label_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            cm[label_idx[t]][label_idx[p]] += 1

    p_o = sum(cm[i][i] for i in range(k)) / n

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

label_idx = {l: i for i, l in enumerate(LABELS)}
k = len(LABELS)
cm = [[0] * k for _ in range(k)]
for t, p in zip(y_human, y_llm):
    if t in label_idx and p in label_idx:
        cm[label_idx[t]][label_idx[p]] += 1

def format_cm_text(cm: list, labels: list) -> str:
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


print(SEP)
print("EVALUATION RESULTS")
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

disagree = merged[merged["label_human"] != merged["label_llm"]].copy()
print(SEP)
print(f"DISAGREEMENTS: {len(disagree)} / {len(merged)} articles")
print(SEP)

if len(disagree) > 0:
    disagree["pair"] = disagree.apply(
        lambda r: f"Human={r['label_human']} vs LLM={r['label_llm']}", axis=1
    )
    pair_counts = Counter(disagree["pair"])
    print("\nMost common disagreement patterns:")
    for pair, cnt in pair_counts.most_common():
        pct = cnt / len(merged) * 100
        print(f"  {pair:<40s} {cnt:3d} articles  ({pct:.1f}%)")

    print(f"\n--- Detail: {len(disagree)} disagreements ---\n")
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

print(SEP)
PASS = True
KAPPA_THRESHOLD = 0.6

if kappa >= KAPPA_THRESHOLD:
    print(f"PASS: Cohen's kappa = {kappa:.4f} >= {KAPPA_THRESHOLD}")
else:
    print(f"FAIL: Cohen's kappa = {kappa:.4f} < {KAPPA_THRESHOLD}")
    PASS = False

    if len(disagree) > 0:
        top_pair, top_cnt = pair_counts.most_common(1)[0]
        print(f"\n  Largest disagreement group: {top_pair} ({top_cnt} articles)")
        h_lbl = top_pair.split("Human=")[1].split(" vs")[0]
        l_lbl = top_pair.split("LLM=")[1]
        print()
        print("  Analysis and guideline improvement suggestions:")
        suggestions = {
            ("POSITIVE", "NEUTRAL"): (
                "LLM NEUTRAL when human says POSITIVE. "
                "Likely cause: article has POSITIVE signals but lacks specific numbers. "
                "Suggestion: Add 'POSITIVE without absolute numbers' examples to Section 7. "
                "Consider lowering specificity threshold for contract/IPO news."
            ),
            ("NEUTRAL", "POSITIVE"): (
                "LLM POSITIVE when human says NEUTRAL. "
                "Likely cause: LLM influenced by positive language in PR/awards. "
                "Suggestion: Add more pure PR examples to Case 7, emphasize 'no financial data -> NEUTRAL'."
            ),
            ("NEGATIVE", "NEUTRAL"): (
                "LLM NEUTRAL when human says NEGATIVE. "
                "Likely cause: mild warning, indirect margin cut. "
                "Suggestion: Clarify Case 5 — indirect warnings also count as NEGATIVE."
            ),
            ("NEUTRAL", "NEGATIVE"): (
                "LLM NEGATIVE when human says NEUTRAL. "
                "Likely cause: LLM oversensitive to negative language in macro news. "
                "Suggestion: Emphasize Case 9 (macro news) and 'when in doubt -> NEUTRAL' rule."
            ),
            ("POSITIVE", "NEGATIVE"): (
                "Maximum disagreement. "
                "Likely cause: article has both positive and negative signals. "
                "Suggestion: Add 'conflicting signals -> NEUTRAL' rule to Section 6."
            ),
            ("NEGATIVE", "POSITIVE"): (
                "Maximum disagreement. "
                "Likely cause: LLM misses hidden negative signal. "
                "Suggestion: Add example of insider selling alongside good news to Case 2."
            ),
        }
        key = (h_lbl, l_lbl)
        suggestion = suggestions.get(
            key,
            f"Review {top_cnt} cases of {top_pair} and add examples to the guideline."
        )
        for line in suggestion.split(". "):
            if line.strip():
                print(f"    • {line.strip()}.")
print()

DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)

from datetime import datetime

with open(DOCS_OUT, "w", encoding="utf-8") as fh:
    def w(s=""):
        fh.write(s + "\n")

    w("# Label Quality Report")
    w()
    w(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    w(f"**Gold set:** `data/interim/gold_seed.csv` ({len(merged)} matched articles)  ")
    w(f"**Machine labels:** `data/interim/labeled_auto.jsonl`")
    w()
    w("---")
    w()

    w("## Summary metrics")
    w()
    w("| Metric | Value |")
    w("|--------|--------:|")
    w(f"| N (matched) | {len(merged)} |")
    w(f"| Accuracy | {acc:.4f} ({acc*100:.1f}%) |")
    w(f"| **Cohen's kappa** | **{kappa:.4f}** |")
    w(f"| Macro-F1 | {macro_f1:.4f} |")
    w()

    w("## Per-class metrics")
    w()
    w("| Label | Precision | Recall | F1 |")
    w("|------|----------:|-------:|---:|")
    for lbl in LABELS:
        v = per_class[lbl]
        w(f"| {lbl} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    w()

    w("## Confusion Matrix")
    w()
    w("> Rows = human labels, Columns = machine labels")
    w()
    w("| Human \\ LLM | " + " | ".join(f"**{l}**" for l in LABELS) + " |")
    w("|-------------|" + "|".join(["------:"] * k) + "|")
    for i, row_label in enumerate(LABELS):
        row_cells = " | ".join(str(cm[i][j]) for j in range(k))
        w(f"| **{row_label}** | {row_cells} |")
    w()

    human_cnt = Counter(y_human)
    llm_cnt   = Counter(y_llm)
    w("## Label distribution")
    w()
    w("| Label | Human (%) | LLM (%) |")
    w("|------|----------:|--------:|")
    for lbl in LABELS:
        h_pct = human_cnt.get(lbl, 0) / len(y_human) * 100
        l_pct = llm_cnt.get(lbl, 0)   / len(y_llm)   * 100
        w(f"| {lbl} | {h_pct:.1f}% | {l_pct:.1f}% |")
    w()

    w("## Disagreement detail")
    w()
    w(f"Total disagreements: **{len(disagree)}** / {len(merged)} ({len(disagree)/len(merged)*100:.1f}%)")
    w()
    if len(disagree) > 0:
        w("### Disagreement groups")
        w()
        w("| Human → LLM | Count | Rate |")
        w("|-------------|-------:|------:|")
        for pair, cnt in pair_counts.most_common():
            w(f"| {pair} | {cnt} | {cnt/len(merged)*100:.1f}% |")
        w()

        w("### Per-case detail")
        w()
        for idx, (_, row) in enumerate(disagree.iterrows(), 1):
            ticker = str(row.get("primary_ticker", "?"))
            title  = str(row.get("title", ""))
            reason = str(row.get("reason", ""))
            conf   = float(row.get("confidence", 0))
            w(f"**[{idx}]** `{ticker}` — {title}  ")
            w(f"- Human: `{row['label_human']}` | LLM: `{row['label_llm']}` (conf={conf:.2f})  ")
            w(f"- LLM reason: *{reason}*")
            w()

    status = "PASS" if PASS else "FAIL"
    w("## Check result")
    w()
    w(f"**{status}**")
    w()
    w(f"- Cohen's kappa = **{kappa:.4f}** (threshold >= {KAPPA_THRESHOLD})")
    w()

    if not PASS and len(disagree) > 0:
        top_pair, top_cnt = pair_counts.most_common(1)[0]
        h_lbl = top_pair.split("Human=")[1].split(" vs")[0]
        l_lbl = top_pair.split("LLM=")[1]
        w("## Guideline improvement suggestions")
        w()
        w(f"> Largest disagreement group: **{top_pair}** ({top_cnt} articles)")
        w()
        suggestion = suggestions.get(
            (h_lbl, l_lbl),
            f"Review {top_cnt} cases of {top_pair} and add examples to the guideline."
        )
        for line in suggestion.split(". "):
            if line.strip():
                w(f"- {line.strip()}.")
        w()
        w("> **Note:** Do not update the guideline automatically — manual review required first.")

print(f"Written: {DOCS_OUT}")
print(SEP)

if not PASS:
    sys.exit(1)