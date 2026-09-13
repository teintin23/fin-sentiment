from __future__ import annotations

import io
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "interim" / "articles_clean.parquet"
NAMES_FILE = ROOT / "data" / "company_names.json"
TICKERS_FILE = ROOT / "data" / "tickers.txt"
OLD_GOLD = ROOT / "data" / "interim" / "gold_seed.csv"

OUT_SCORED = ROOT / "data" / "interim" / "articles_scored.parquet"
OUT_LABEL = ROOT / "data" / "interim" / "to_label_v2.parquet"
OUT_GOLD = ROOT / "data" / "interim" / "gold_seed_v2.csv"
OUT_REPORT = ROOT / "docs" / "ticker_repair_report.md"

RANDOM_STATE = 42
TOTAL_TARGET = 6000
MAX_PER_TICKER = 120
GOLD_SIZE = 200

MIN_SCORE = 55
MIN_MARGIN = 15

CODE_BLACKLIST = {
    "USD", "GDP", "CEO", "FDI", "HSX", "ETF", "IPO", "EPS", "ROE", "ROA",
    "CPI", "FED", "ECB", "COO", "CFO", "CTO", "TPP", "WTO", "VAT", "BOT",
    "PPP", "GMT", "USA", "PMI", "NIM", "CAR", "ESG", "SME", "CCP", "VKS",
    "TOP", "MTV", "KKT", "KCN", "SJC", "ATM", "POS", "API",
    "HNX", "UPC", "GTV", "TTC",
    "EUR", "JPY", "CNY", "KRW", "THB", "SGD", "AUD", "GBP", "HKD", "TWD",
    "BOJ", "PBOC", "IMF", "ADB", "WEF", "OPEC",
    "LNG", "LPG", "CNG", "EVN", "TKV", "PVN", "SCIC", "VEC",
    "FTA", "ODA", "NPL", "ROI", "IRR", "NPV",
    "AGM", "EGM", "MOU", "LOI",
    "SMS", "OTP", "KYC", "AML", "PIN", "NFC", "GPS", "USB", "PDF",
    "TVC", "PGD",
    "UBN", "UBC", "SSC", "VSD",
    "HCM", "TPH", "HAN", "DAN",
    "COD", "FOB", "CIF", "TEU", "MWH", "KWH", "GWH",
}

NEEDS_NAME_CONFIRM = {
    "VND", "GAS", "PAN", "TIN", "CAN", "BAN", "TAN", "SAM", "NAM", "HAI",
    "MAI", "LAN", "VAN", "CAP", "ATA", "ART", "TOP", "NET", "ONE", "SEA",
    "PET", "PHP", "SGD", "ITA", "ASP", "APP", "ACE", "AME", "AMP", "ADC",
    "BOT", "MAC", "MEC", "MHC", "TET", "TVB", "SIP", "SAV", "SGN", "VAT",
    "VIP", "VLC",
}

GENERIC_ALIASES = {
    "nam a", "hop nhat", "tien bo", "thanh cong", "phat trien", "dau tu",
    "xay dung", "thuong mai", "dich vu", "san xuat", "viet nam", "ha noi",
    "sai gon", "mien bac", "mien nam", "mien trung", "toan cau", "quoc te",
    "cong nghe", "van tai", "bat dong san", "du lich", "khoang san",
    "nong nghiep", "thuy san", "dien luc", "xuat nhap khau", "tap doan",
    "an phat", "hoa binh", "binh minh", "dai duong", "song da", "song hong",
    "truong thanh", "tien phong", "hong ha", "thang long", "bien dong",
    "dong a", "tay do", "hai phong", "da nang", "can tho", "vinh long",
}

ROUNDUP_PATTERNS = [
    r"t[iy]̉? ?gi[áa]", r"gi[áa] ?USD", r"USD ng[âa]n h[àa]ng",
    r"gi[áa] ?v[àa]ng", r"v[àa]ng nh[ẫa]n", r"v[àa]ng mi[ếe]ng",
    r"l[ãa]i su[ấa]t (huy [đd][ộo]ng|ti[ếe]t ki[ệe]m|h[ôo]m nay|ng[âa]n h[àa]ng)",
    r"l[ịi]ch ch[ốo]t quy[ềe]n", r"l[ăa]n ch[ốo]t",
    r"l[ịi]ch s[ựu] ki[ệe]n v[àa] tin v[ắa]n",
    r"kh[ốo]i ngo[ạa]i", r"t[ựu] doanh",
    r"nh[ậa]n [đd][ịi]nh (ch[ứa]ng kho[áa]n|th[ịi] tr[ưu][ơờ]ng)",
    r"g[óo]c nh[ìi]n CTCK", r"khuy[ếe]n ngh[ịi] (mua|b[áa]n|c[ổo] phi[ếe]u)",
    r"VN-?Index", r"ph[iê]en (s[áa]ng|chi[ềe]u|giao d[ịi]ch) ng[àa]y",
    r"th[ịi] tr[ưu][ơờ]ng ch[ứa]ng kho[áa]n (h[ôo]m nay|tu[ầa]n)",
    r"c[ậa]p nh[ậa]t s[ốo] li[ệe]u CTCK",
    r"top \d+ ", r"danh s[áa]ch \d+",
]
ROUNDUP_RE = re.compile("|".join(ROUNDUP_PATTERNS), re.IGNORECASE)

AD_PATTERNS = [
    r"[đd][ồo]ng h[àa]nh c[ùu]ng", r"tri [âa]n kh[áa]ch h[àa]ng",
    r"[ưu]u [đd][ãa]i (l[ơớ]n|h[ấa]p d[ẫa]n|[đd][ặa]c bi[ệe]t)",
    r"vinh danh", r"gi[ảa]i th[ưu][ơở]ng", r"[đd][ưu][ợo]c b[ìi]nh ch[ọo]n",
    r"l[ọo]t top \d+", r"ra m[ắa]t (s[ảa]n ph[ẩa]m|d[ịi]ch v[ụu]|b[ộo] nh[ậa]n di[ệe]n)",
    r"c[ơơ] h[ộo]i tr[úu]ng", r"qu[àa] t[ặa]ng", r"[đd][ăa]ng k[ýy] ngay",
]
AD_RE = re.compile("|".join(AD_PATTERNS), re.IGNORECASE)

CURRENCY_CONTEXT = re.compile(
    r"t[ỷy] gi[áa]|USD|ngo[ạa]i t[ệe]|[đd][ồo]ng b[ạa]c xanh|EUR|Yen|Nh[âa]n d[âa]n t[ệe]",
    re.IGNORECASE,
)

WS = re.compile(r"\s+")


def fold(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return WS.sub(" ", s).strip()


print("=" * 68)
print("Loading ...")
df = pd.read_parquet(IN_FILE)
print(f"  articles_clean : {len(df):,} rows")

whitelist = {
    t.strip().upper()
    for t in re.split(r"[\s,;]+", TICKERS_FILE.read_text(encoding="utf-8"))
    if t.strip()
}
print(f"  ticker universe: {len(whitelist):,}")

raw_names = json.loads(NAMES_FILE.read_text(encoding="utf-8"))

alias_map: dict[str, str] = {}
dropped_alias = 0
for code, aliases in raw_names.items():
    if code not in whitelist:
        continue
    for a in aliases:
        fa = fold(a)
        if len(fa) < 6 or fa in GENERIC_ALIASES:
            dropped_alias += 1
            continue
        if fa in alias_map and alias_map[fa] != code:
            alias_map[fa] = "__AMBIG__"
        else:
            alias_map.setdefault(fa, code)
alias_map = {k: v for k, v in alias_map.items() if v != "__AMBIG__"}
print(f"  aliases kept   : {len(alias_map):,}  (dropped {dropped_alias:,} generic/short)")

ALIAS_SORTED = sorted(alias_map.items(), key=lambda p: -len(p[0]))

CODE_RE = re.compile(r"\b([A-Z]{3})\b")


def codes_in(text: str) -> dict[str, int]:
    out = {}
    for m in CODE_RE.finditer(text or ""):
        c = m.group(1)
        if c in CODE_BLACKLIST or c not in whitelist:
            continue
        out.setdefault(c, m.start())
    return out


def aliases_in(folded: str) -> dict[str, int]:
    out = {}
    if not folded:
        return out
    padded = f" {folded} "
    for fa, code in ALIAS_SORTED:
        pos = padded.find(f" {fa} ")
        if pos == -1:
            pos = padded.find(f" {fa}")
            if pos == -1:
                continue
            nxt = pos + 1 + len(fa)
            if nxt < len(padded) and padded[nxt] not in " ":
                continue
        if code not in out:
            out[code] = pos
    return out


def score_row(row) -> tuple:
    title = row["title"] if isinstance(row["title"], str) else ""
    sapo = row["sapo"] if isinstance(row["sapo"], str) else ""
    body = (row["body"] if isinstance(row["body"], str) else "")[:4000]
    head = f"{title} {sapo}"
    f_title, f_sapo, f_body = fold(title), fold(sapo), fold(body)

    _w = row["tickers_widget"]
    widget = set(list(_w)) if _w is not None and len(_w) > 0 else set()

    c_title = codes_in(title)
    c_sapo = codes_in(sapo)
    c_body = codes_in(body)
    a_title = aliases_in(f_title)
    a_sapo = aliases_in(f_sapo)
    a_body = aliases_in(f_body)

    is_roundup = bool(ROUNDUP_RE.search(head))
    is_ad = bool(AD_RE.search(head))
    currency_ctx = bool(CURRENCY_CONTEXT.search(head))

    cands = (
        set(c_title) | set(c_sapo) | set(a_title) | set(a_sapo)
        | set(c_body) | set(a_body) | widget
    )

    scores: dict[str, float] = defaultdict(float)
    for t in cands:
        s = 0.0
        name_confirmed = t in a_title or t in a_sapo or t in a_body

        if t in c_title:
            s += 100 - min(30, c_title[t] * 0.4)
        if t in a_title:
            s += 90 - min(25, a_title[t] * 0.3)
        if t in c_sapo:
            s += 55
        if t in a_sapo:
            s += 50
        if t in widget:
            s += 30
        if t in c_body:
            s += max(0.0, 22 - c_body[t] / 120)
        if t in a_body:
            s += max(0.0, 20 - a_body[t] / 120)

        if t in NEEDS_NAME_CONFIRM and not name_confirmed:
            s = 0.0
        if t == "VND" and currency_ctx and not name_confirmed:
            s = 0.0

        if s > 0:
            scores[t] = s

    if not scores:
        return (None, 0.0, 0.0, 0, is_roundup, is_ad, "no_candidate")

    ranked = sorted(scores.items(), key=lambda p: -p[1])
    top, top_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_s - second_s

    reason = "ok"
    if top_s < MIN_SCORE:
        reason = "low_score"
    elif margin < MIN_MARGIN:
        reason = "ambiguous"
    elif is_roundup and top_s < 120:
        reason = "roundup"
    elif is_ad and top_s < 120:
        reason = "advertorial"

    return (top, round(top_s, 1), round(margin, 1), len(ranked),
            is_roundup, is_ad, reason)


print("\nScoring articles (~1-2 min) ...")
res = df.apply(score_row, axis=1, result_type="expand")
res.columns = ["pt_new", "pt_score", "pt_margin", "n_cand",
               "is_roundup", "is_ad", "pt_reason"]
df = pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)

df["pt_ok"] = df["pt_reason"] == "ok"


def _in_head(r):
    pt = r["pt_new"]
    if not isinstance(pt, str) or not pt:
        return False
    head = f"{r['title'] if isinstance(r['title'], str) else ''} {r['sapo'] if isinstance(r['sapo'], str) else ''}"
    if pt in head:
        return True
    return pt in aliases_in(fold(head))


df["ticker_in_head"] = df.apply(_in_head, axis=1)


def _text_input(r):
    pt = r["pt_new"] if isinstance(r["pt_new"], str) and r["pt_new"] else "MACRO"
    t = r["title"] if isinstance(r["title"], str) else ""
    sp = r["sapo"] if isinstance(r["sapo"], str) else ""
    return WS.sub(" ", f"[{pt}] {t}. {sp}").strip()


df["text_input"] = df.apply(_text_input, axis=1)

old_has = df["primary_ticker"].notna()
print("\n" + "=" * 68)
print("RESULTS")
print("=" * 68)
print(f"  Old  : {old_has.sum():,} articles with primary_ticker")
print(f"  New  : {df['pt_ok'].sum():,} articles with qualifying primary_ticker")
print("\n  Rejection reasons:")
print(df["pt_reason"].value_counts().to_string())

both = df[old_has & df["pt_ok"]]
agree = (both["primary_ticker"] == both["pt_new"]).mean()
print(f"\n  Agreement with old pipeline (on articles where both have ticker): {agree:.1%}")
print(f"  -> {(1-agree)*len(both):,.0f} articles changed ticker")

print(f"\n  Ticker present in title+sapo: {df.loc[df['pt_ok'],'ticker_in_head'].mean():.1%}")

OUT_SCORED.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_SCORED, index=False, engine="pyarrow")
print(f"\n  Written: {OUT_SCORED}")

pool = df[df["pt_ok"]].copy()
print(f"\nEligible pool: {len(pool):,}"
      f"  (ticker in title+sapo: {pool['ticker_in_head'].mean():.1%})")

parts = []
for t, g in pool.groupby("pt_new", sort=False):
    parts.append(g.sample(n=min(len(g), MAX_PER_TICKER), random_state=RANDOM_STATE))
sampled = pd.concat(parts, ignore_index=True)
if len(sampled) > TOTAL_TARGET:
    sampled = sampled.sample(n=TOTAL_TARGET, random_state=RANDOM_STATE)
sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

KEEP = ["id", "url", "date", "time", "datetime", "pt_new", "title", "sapo",
        "text", "text_input", "pt_score", "pt_margin", "n_cand",
        "ticker_in_head", "is_roundup", "is_ad", "tickers_any"]
to_label = sampled[[c for c in KEEP if c in sampled.columns]].rename(
    columns={"pt_new": "primary_ticker"}
)
to_label.to_parquet(OUT_LABEL, index=False, engine="pyarrow")
print(f"  to_label_v2: {len(to_label):,} articles, {to_label['primary_ticker'].nunique():,} tickers")
print(f"  Written: {OUT_LABEL}")

old_gold = pd.read_csv(OLD_GOLD, encoding="utf-8-sig")
old_gold["id"] = old_gold["id"].astype(str)
old_map = old_gold.set_index("id")[["primary_ticker", "label", "note"]]

carry = to_label[to_label["id"].astype(str).isin(old_map.index)].copy()
carry["_old_tk"] = carry["id"].astype(str).map(old_map["primary_ticker"])
carry = carry[carry["_old_tk"] == carry["primary_ticker"]].drop(columns=["_old_tk"])
rest = to_label[~to_label["id"].isin(carry["id"])]
n_new = max(0, GOLD_SIZE - len(carry))
gold = pd.concat(
    [carry, rest.sample(n=min(n_new, len(rest)), random_state=RANDOM_STATE)],
    ignore_index=True,
).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
gold["label"] = ""
gold["note"] = ""
gold["reused"] = False

reused = 0
for i, r in gold.iterrows():
    rid = str(r["id"])
    if rid in old_map.index:
        old_row = old_map.loc[rid]
        if old_row["primary_ticker"] == r["primary_ticker"]:
            gold.at[i, "label"] = old_row["label"]
            gold.at[i, "note"] = old_row["note"]
            gold.at[i, "reused"] = True
            reused += 1

survive = old_gold[old_gold["id"].isin(to_label["id"])]
same_tk = survive.merge(
    to_label[["id", "primary_ticker"]], on="id", suffixes=("_old", "_new")
)
n_same = (same_tk["primary_ticker_old"] == same_tk["primary_ticker_new"]).sum()

gold_cols = ["id", "date", "primary_ticker", "ticker_in_head", "title", "sapo",
             "text_input", "url", "label", "note", "reused"]
gold[[c for c in gold_cols if c in gold.columns]].to_csv(
    OUT_GOLD, index=False, encoding="utf-8-sig"
)
print(f"\n  gold_seed_v2: {len(gold)} articles, {reused} labels reused from old gold seed")
print(f"  Of 150 old labels: {len(survive)} still in new pool, "
      f"{n_same} with same ticker -> labels remain valid")
print(f"  Written: {OUT_GOLD}")

changed = both[both["primary_ticker"] != both["pt_new"]].head(25)
lines = [
    "# Ticker repair report",
    "",
    f"- Articles with primary_ticker (old): **{old_has.sum():,}**",
    f"- Articles with qualifying primary_ticker (new): **{df['pt_ok'].sum():,}**",
    f"- Agreement between pipelines: **{agree:.1%}**",
    f"- Ticker present in title+sapo (new): **{df.loc[df['pt_ok'],'ticker_in_head'].mean():.1%}** "
    f"(old: ~68%)",
    "",
    "## Rejection reasons",
    "",
    "| Reason | Count |",
    "|---|---|",
]
for k, v in df["pt_reason"].value_counts().items():
    lines.append(f"| {k} | {v:,} |")
lines += [
    "",
    "## Examples of changed ticker",
    "",
    "| Old | New | Title |",
    "|---|---|---|",
]
for _, r in changed.iterrows():
    t = str(r["title"]).replace("|", "/")[:95]
    lines.append(f"| {r['primary_ticker']} | {r['pt_new']} | {t} |")

OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"  Written: {OUT_REPORT}")
print("=" * 68)
