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

    dis["suspicion"] = (
        dis["confidence"]
        + 0.35 * (dis["rule"] == dis["label_machine"])
        + 0.25 * (dis["label_source"] == "claude_manual")
        + 0.20 * (~dis["reused"].astype(bool))
    ).round(3)

    dis["label_suggest"] = dis["label_machine"]
    dis["label_final"] = ""
    dis = dis.sort_values("suspicion", ascending=False)

    cols = ["id", "primary_ticker", "date", "reused", "title", "sapo", "url",
            "label_human", "label_machine", "rule", "confidence", "lex_net",
            "suspicion", "label_suggest", "label_final"]
    dis[cols].to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"Disagreements: {len(dis)}/{len(m)} ({len(dis)/len(m):.1%})")
    print(f"  new labels (reused=False): {int((~dis['reused'].astype(bool)).sum())}")
    print(f"  old labels (reused=True) : {int(dis['reused'].astype(bool).sum())}")
    print("\nTop disagreement patterns:")
    pat = (dis["label_human"] + " -> " + dis["label_machine"]).value_counts()
    for k, v in pat.items():
        print(f"  human={k.split(' -> ')[0]:9s} machine={k.split(' -> ')[1]:9s} {v:3d}")
    print(f"\nWritten {OUT}")
    print("Open file, read title/sapo, fill POSITIVE/NEUTRAL/NEGATIVE in label_final")
    print("(leave blank = keep current human label), then run:")
    print(f"  python src/audit_gold.py --apply {OUT.name}")
    return 0


def apply(fname: str) -> int:
    path = ROOT / "data" / "interim" / fname if not Path(fname).is_absolute() else Path(fname)
    rec = pd.read_csv(path, encoding="utf-8-sig")
    rec = rec[rec["label_final"].isin(LABELS)]
    if rec.empty:
        print("No rows with label_final filled. No changes.")
        return 0

    gold = pd.read_csv(GOLD, encoding="utf-8-sig")
    fix = dict(zip(rec["id"].astype(str), rec["label_final"]))
    before = gold["label"].copy()
    gold["label"] = [fix.get(str(i), l) for i, l in zip(gold["id"], gold["label"])]
    n = int((before.fillna("") != gold["label"].fillna("")).sum())

    bak = GOLD.with_suffix(".csv.bak")
    pd.read_csv(GOLD, encoding="utf-8-sig").to_csv(bak, index=False, encoding="utf-8-sig")
    gold.to_csv(GOLD, index=False, encoding="utf-8-sig")
    print(f"Updated {n} labels in gold_seed_v2.csv (backup saved as {bak.name})")
    print("Run: python src/eval_labels.py")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", metavar="FILE", help="load corrected file into gold_seed_v2")
    a = ap.parse_args()
    sys.exit(apply(a.apply) if a.apply else build())
