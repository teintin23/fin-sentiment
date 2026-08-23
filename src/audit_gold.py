"""
audit_gold.py
-------------
Soát lại nhãn người trong gold_seed_v2.csv tại những chỗ bất đồng với nhãn máy.

Lý do tồn tại: kappa giữa người và máy tụt xuống 0.55, nhưng tách theo cột
``reused`` thì thấy phần nhãn cũ đạt 0.69 còn phần nhãn mới chỉ 0.36 và có tới
70% là POSITIVE. Đó là dấu hiệu trôi tiêu chí trong phiên gán nhãn gần nhất chứ
không phải máy gán sai. Script này lọc ra đúng những dòng cần đọc lại.

Đầu ra: data/interim/gold_recheck.csv, sắp xếp theo mức đáng ngờ giảm dần.
Cột `label_suggest` là gợi ý, KHÔNG tự ghi đè. Người quyết định.

Usage:
    python src/audit_gold.py                 # sinh file soát lại
    python src/audit_gold.py --apply FILE    # nạp lại file đã sửa vào gold_seed_v2
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexicon_vi_fin import rule_label, score_text  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "interim" / "gold_seed_v2.csv"
AUTO = ROOT / "data" / "interim" / "labeled_auto.jsonl"
OUT = ROOT / "data" / "interim" / "gold_recheck.csv"
LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


def build() -> int:
    gold = pd.read_csv(GOLD, encoding="utf-8-sig")
    g = gold[gold["label"].isin(LABELS)].copy()
    llm = pd.DataFrame([json.loads(l) for l in AUTO.open(encoding="utf-8") if l.strip()])
    m = g.merge(llm[["id", "label", "confidence", "label_source"]],
                on="id", suffixes=("_human", "_machine"))

    dis = m[m["label_human"] != m["label_machine"]].copy()
    dis["rule"] = dis["text_input"].map(lambda t: rule_label(str(t)) or "")
    dis["lex_net"] = dis["text_input"].map(lambda t: round(score_text(str(t))["lex_net"], 2))

    # mức đáng ngờ: máy càng chắc chắn + luật từ điển cũng đồng ý với máy
    dis["nghi_ngo"] = (
        dis["confidence"]
        + 0.35 * (dis["rule"] == dis["label_machine"])
        + 0.25 * (dis["label_source"] == "claude_manual")
        + 0.20 * (~dis["reused"].astype(bool))       # nhãn mới trôi nhiều hơn
    ).round(3)

    dis["label_suggest"] = dis["label_machine"]
    dis["label_final"] = ""      # cột để người điền
    dis = dis.sort_values("nghi_ngo", ascending=False)

    cols = ["id", "primary_ticker", "date", "reused", "title", "sapo", "url",
            "label_human", "label_machine", "rule", "confidence", "lex_net",
            "nghi_ngo", "label_suggest", "label_final"]
    dis[cols].to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"Bat dong: {len(dis)}/{len(m)} bai  ({len(dis)/len(m):.1%})")
    print(f"  trong do nhan MOI (reused=False): {int((~dis['reused'].astype(bool)).sum())}")
    print(f"  trong do nhan CU  (reused=True) : {int(dis['reused'].astype(bool).sum())}")
    print("\nKieu bat dong pho bien:")
    pat = (dis["label_human"] + " -> " + dis["label_machine"]).value_counts()
    for k, v in pat.items():
        print(f"  nguoi={k.split(' -> ')[0]:9s} may={k.split(' -> ')[1]:9s} {v:3d} bai")
    print(f"\nDa ghi {OUT}")
    print("Mo file, doc cot title/sapo, dien POSITIVE/NEUTRAL/NEGATIVE vao cot label_final")
    print("(de trong = giu nguyen nhan nguoi hien tai), roi chay:")
    print(f"  python src/audit_gold.py --apply {OUT.name}")
    return 0


def apply(fname: str) -> int:
    path = ROOT / "data" / "interim" / fname if not Path(fname).is_absolute() else Path(fname)
    rec = pd.read_csv(path, encoding="utf-8-sig")
    rec = rec[rec["label_final"].isin(LABELS)]
    if rec.empty:
        print("Khong co dong nao dien label_final. Khong thay doi gi.")
        return 0

    gold = pd.read_csv(GOLD, encoding="utf-8-sig")
    fix = dict(zip(rec["id"].astype(str), rec["label_final"]))
    before = gold["label"].copy()
    gold["label"] = [fix.get(str(i), l) for i, l in zip(gold["id"], gold["label"])]
    n = int((before.fillna("") != gold["label"].fillna("")).sum())

    bak = GOLD.with_suffix(".csv.bak")
    pd.read_csv(GOLD, encoding="utf-8-sig").to_csv(bak, index=False, encoding="utf-8-sig")
    gold.to_csv(GOLD, index=False, encoding="utf-8-sig")
    print(f"Da cap nhat {n} nhan trong gold_seed_v2.csv (ban cu luu o {bak.name})")
    print("Chay lai: python src/eval_labels.py")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", metavar="FILE", help="nap file da sua vao gold_seed_v2")
    a = ap.parse_args()
    sys.exit(apply(a.apply) if a.apply else build())
