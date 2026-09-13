import io
import sys
import re
import json
import unicodedata
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

MANUAL_DICT: dict = {
    "VCB": ["Vietcombank"],
    "CTG": ["VietinBank"],
    "BID": ["BIDV"],
    "TCB": ["Techcombank"],
    "MBB": ["MBBank", "MB Bank"],
    "VPB": ["VPBank"],
    "STB": ["Sacombank"],
    "HDB": ["HDBank"],
    "TPB": ["TPBank"],
    "EIB": ["Eximbank"],
    "SSB": ["SeABank"],
    "VNM": ["Vinamilk"],
    "SAB": ["Sabeco"],
    "VJC": ["Vietjet"],
    "MWG": ["The Gioi Di Dong", "Thế Giới Di Động"],
    "HPG": ["Hoà Phát", "Hoa Phat"],
    "HSG": ["Hoa Sen"],
    "GAS": ["PV GAS"],
    "PLX": ["Petrolimex"],
    "POW": ["PV Power"],
    "VNZ": ["VNG"],
    "HVN": ["Vietnam Airlines"],
    "VCG": ["Vinaconex"],
    "GEX": ["Gelex"],
    "GEE": ["GELEX Electric"],
    "HDG": ["Hà Đô", "Ha Do"],
    "VND": ["VNDirect"],
    "POM": ["Thép Pomina", "Thep Pomina", "Pomina"],
    "QCG": ["Quốc Cường Gia Lai", "Quoc Cuong Gia Lai"],
    "HAG": ["Hoàng Anh Gia Lai", "Hoang Anh Gia Lai", "HAGL"],
    "NVL": ["Novaland"],
    "VHM": ["Vinhomes"],
    "VIC": ["Vingroup"],
    "VRE": ["Vincom Retail", "Vincom"],
    "FRT": ["FPT Retail"],
}

LEGAL_PREFIXES = [
    r"công ty cổ phần",
    r"ctcp",
    r"ngân hàng thương mại cổ phần",
    r"ngân hàng tmcp",
    r"ngân hàng",
    r"tổng công ty cổ phần",
    r"tổng công ty",
    r"tập đoàn",
    r"công ty tnhh",
    r"công ty",
    r"tnhh",
]

GENERIC_LOWER = {
    "đầu tư",
    "phát triển",
    "xây dựng",
    "thương mại",
    "dịch vụ",
    "việt nam",
    "viet nam",
    "vietnam",
    "chứng khoán",
    "bất động sản",
    "bat dong san",
    "sản xuất",
    "kinh doanh",
    "tư vấn",
    "quản lý",
    "đầu tư phát triển",
    "đầu tư xây dựng",
    "đầu tư thương mại",
    "thương mại dịch vụ",
    "xây dựng thương mại",
}


def normalize_vi(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def remove_legal_prefix(name: str) -> str:
    s = unicodedata.normalize("NFC", name.strip().lower())
    for prefix in LEGAL_PREFIXES:
        pattern = r"^" + prefix + r"[\s\-\u2013\u2014:]*"
        s_new = re.sub(pattern, "", s, flags=re.IGNORECASE).strip()
        if s_new != s:
            s = s_new
            break
    return s.strip()


def title_case_vi(text: str) -> str:
    return " ".join(w.capitalize() for w in text.split())


def is_generic(alias_lower: str) -> bool:
    return alias_lower in GENERIC_LOWER


def make_accent_variants(alias: str) -> list:
    variants = [alias]
    no_accent = unicodedata.normalize("NFD", alias)
    no_accent = "".join(c for c in no_accent if unicodedata.category(c) != "Mn")
    no_accent = unicodedata.normalize("NFC", no_accent)
    if no_accent != alias:
        variants.append(no_accent)
    return variants


def build_aliases_from_name(organ_name: str) -> list:
    if not isinstance(organ_name, str) or not organ_name.strip():
        return []

    name = normalize_vi(organ_name)
    core = remove_legal_prefix(name)

    if not core or len(core) < 4:
        return []
    if len(core.split()) > 6:
        return []
    if is_generic(core):
        return []

    alias = title_case_vi(core)
    return make_accent_variants(alias)


def fetch_listing_df() -> pd.DataFrame:
    from vnstock.api.listing import Listing

    listing = Listing()

    methods_to_try = [
        ("all_symbols", {}),
        ("symbols_by_exchange", {"exchange": "ALL"}),
        ("symbols_by_exchange", {"exchange": "HOSE"}),
        ("symbols_by_industries", {}),
        ("symbols_by_group", {"group": "VN30"}),
    ]

    for method_name, kwargs in methods_to_try:
        if not hasattr(listing, method_name):
            print(f"[SKIP] Method '{method_name}' does not exist.")
            continue

        label = method_name + (f"({kwargs})" if kwargs else "()")
        print(f"\n[TRY] Listing().{label} ...")
        try:
            method = getattr(listing, method_name)
            df = method(**kwargs) if kwargs else method()

            if not isinstance(df, pd.DataFrame):
                print(f"  -> Not a DataFrame (type={type(df)}), skipping.")
                continue

            print(f"  OK  Rows={len(df)}, Columns={list(df.columns)}")

            if len(df) > 100:
                print(f"  => Using: {label}  ({len(df)} rows)\n")
                return df
            else:
                print(f"  -> Only {len(df)} rows, trying next method...")

        except Exception:
            import traceback
            print(f"  ERROR '{method_name}':")
            traceback.print_exc()

    raise RuntimeError(
        "No method returned a DataFrame with >100 rows.\n"
        "Check network connection or vnstock API."
    )


def detect_columns(df: pd.DataFrame):
    ticker_candidates = ["symbol", "ticker", "code", "stockCode", "stock_code"]
    name_candidates = [
        "organ_name", "organName",
        "company_name", "companyName",
        "short_name", "shortName",
        "name",
    ]

    ticker_col = next((c for c in ticker_candidates if c in df.columns), None)
    name_col = next((c for c in name_candidates if c in df.columns), None)

    if ticker_col is None:
        raise ValueError(
            f"Cannot find ticker column in DataFrame. "
            f"Columns: {list(df.columns)}"
        )

    print(f"[DETECT] ticker_col='{ticker_col}', name_col='{name_col}'")
    return ticker_col, name_col


def main() -> None:
    print("=" * 60)
    print("  BUILD TICKER DICT  (vnstock API)")
    print("=" * 60)

    df = fetch_listing_df()

    raw_path = DATA_DIR / "listing_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] listing_raw.csv  ({len(df)} rows) -> {raw_path}")

    ticker_col, name_col = detect_columns(df)

    tickers_raw = df[ticker_col].dropna().astype(str).str.strip().str.upper()
    valid_tickers = sorted({t for t in tickers_raw if re.match(r"^[A-Z]{3}$", t)})

    ticker_path = DATA_DIR / "tickers.txt"
    ticker_path.write_text("\n".join(valid_tickers) + "\n", encoding="utf-8")
    print(f"[SAVE] tickers.txt  ({len(valid_tickers)} tickers) -> {ticker_path}")

    alias_to_tickers: dict = {}
    ticker_to_aliases: dict = {}

    if name_col:
        for _, row in df.iterrows():
            ticker = str(row[ticker_col]).strip().upper()
            if not re.match(r"^[A-Z]{3}$", ticker):
                continue
            organ_name = row.get(name_col, "")
            aliases = build_aliases_from_name(str(organ_name))
            for a in aliases:
                alias_to_tickers.setdefault(a.lower(), []).append(ticker)
                ticker_to_aliases.setdefault(ticker, set()).add(a)

        ambiguous = {a for a, ts in alias_to_tickers.items() if len(ts) > 1}
        print(f"[INFO] Aliases removed (ambiguous, matched multiple tickers): {len(ambiguous)}")

        for ticker in ticker_to_aliases:
            ticker_to_aliases[ticker] = {
                a for a in ticker_to_aliases[ticker]
                if a.lower() not in ambiguous
            }

    company_names: dict = {}

    for ticker in valid_tickers:
        aliases: set = set(ticker_to_aliases.get(ticker, set()))
        if ticker in MANUAL_DICT:
            for manual_alias in MANUAL_DICT[ticker]:
                for variant in make_accent_variants(manual_alias):
                    aliases.add(variant)
        if aliases:
            company_names[ticker] = sorted(aliases)

    json_path = DATA_DIR / "company_names.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(company_names, f, indent=1, ensure_ascii=False)
    print(f"[SAVE] company_names.json  ({len(company_names)} tickers) -> {json_path}")

    print()
    print("=" * 60)
    print(f"  TICKERS : {len(valid_tickers):>5}  (PASS if >= 500)")
    print(f"  COMPANIES: {len(company_names):>5}  (PASS if >= 300)")
    print("=" * 60)

    if len(valid_tickers) >= 500:
        print("[PASS] tickers.txt OK")
    else:
        print(f"[FAIL] tickers.txt only has {len(valid_tickers)} tickers")

    if len(company_names) >= 300:
        print("[PASS] company_names.json OK")
    else:
        print(f"[FAIL] company_names.json only has {len(company_names)} tickers")


if __name__ == "__main__":
    main()