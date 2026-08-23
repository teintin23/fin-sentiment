"""
build_ticker_dict.py
--------------------
Sinh whitelist mã cổ phiếu (data/tickers.txt) và từ điển tên công ty
(data/company_names.json) từ vnstock.

Yêu cầu: vnstock >= 4.0.6 (dùng vnstock.api mới, không dùng class Vnstock cũ)
"""

import io
import sys
import re
import json
import unicodedata
from pathlib import Path

import pandas as pd

# --- Đảm bảo stdout/stderr là UTF-8 (Windows) --------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Manual brand aliases ----------------------------------------------------
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
    "MWG": ["The Gioi Di Dong", "Th\u1ebf Gi\u1edbi Di \u0110\u1ed9ng"],
    "HPG": ["Ho\u00e0 Ph\u00e1t", "Hoa Phat"],
    "HSG": ["Hoa Sen"],
    "GAS": ["PV GAS"],
    "PLX": ["Petrolimex"],
    "POW": ["PV Power"],
    "VNZ": ["VNG"],
    "HVN": ["Vietnam Airlines"],
    "VCG": ["Vinaconex"],
    "GEX": ["Gelex"],
    "GEE": ["GELEX Electric"],
    "HDG": ["H\u00e0 \u0110\u00f4", "Ha Do"],
    "VND": ["VNDirect"],
    "POM": ["Th\u00e9p Pomina", "Thep Pomina", "Pomina"],
    "QCG": ["Qu\u1ed1c C\u01b0\u1eddng Gia Lai", "Quoc Cuong Gia Lai"],
    "HAG": ["Ho\u00e0ng Anh Gia Lai", "Hoang Anh Gia Lai", "HAGL"],
    "NVL": ["Novaland"],
    "VHM": ["Vinhomes"],
    "VIC": ["Vingroup"],
    "VRE": ["Vincom Retail", "Vincom"],
    "FRT": ["FPT Retail"],
}

# --- Tiền tố pháp lý cần bỏ --------------------------------------------------
LEGAL_PREFIXES = [
    r"c\u00f4ng ty c\u1ed5 ph\u1ea7n",
    r"ctcp",
    r"ng\u00e2n h\u00e0ng th\u01b0\u01a1ng m\u1ea1i c\u1ed5 ph\u1ea7n",
    r"ng\u00e2n h\u00e0ng tmcp",
    r"ng\u00e2n h\u00e0ng",
    r"t\u1ed5ng c\u00f4ng ty c\u1ed5 ph\u1ea7n",
    r"t\u1ed5ng c\u00f4ng ty",
    r"t\u1eadp \u0111o\u00e0n",
    r"c\u00f4ng ty tnhh",
    r"c\u00f4ng ty",
    r"tnhh",
]

# --- Cụm từ chung chung cần loại ---------------------------------------------
GENERIC_LOWER = {
    "\u0111\u1ea7u t\u01b0",
    "ph\u00e1t tri\u1ec3n",
    "x\u00e2y d\u1ef1ng",
    "th\u01b0\u01a1ng m\u1ea1i",
    "d\u1ecbch v\u1ee5",
    "vi\u1ec7t nam",
    "viet nam",
    "vietnam",
    "ch\u1ee9ng kho\u00e1n",
    "b\u1ea5t \u0111\u1ed9ng s\u1ea3n",
    "bat dong san",
    "s\u1ea3n xu\u1ea5t",
    "kinh doanh",
    "t\u01b0 v\u1ea5n",
    "qu\u1ea3n l\u00fd",
    "\u0111\u1ea7u t\u01b0 ph\u00e1t tri\u1ec3n",
    "\u0111\u1ea7u t\u01b0 x\u00e2y d\u1ef1ng",
    "\u0111\u1ea7u t\u01b0 th\u01b0\u01a1ng m\u1ea1i",
    "th\u01b0\u01a1ng m\u1ea1i d\u1ecbch v\u1ee5",
    "x\u00e2y d\u1ef1ng th\u01b0\u01a1ng m\u1ea1i",
}


def normalize_vi(text: str) -> str:
    """Chuẩn hoá tiếng Việt: NFC, strip."""
    return unicodedata.normalize("NFC", text.strip())


def remove_legal_prefix(name: str) -> str:
    """Bỏ tiền tố pháp lý ở đầu tên (case-insensitive), trả về lowercase."""
    s = unicodedata.normalize("NFC", name.strip().lower())
    for prefix in LEGAL_PREFIXES:
        pattern = r"^" + prefix + r"[\s\-\u2013\u2014:]*"
        s_new = re.sub(pattern, "", s, flags=re.IGNORECASE).strip()
        if s_new != s:
            s = s_new
            break
    return s.strip()


def title_case_vi(text: str) -> str:
    """Viết hoa chữ đầu mỗi từ."""
    return " ".join(w.capitalize() for w in text.split())


def is_generic(alias_lower: str) -> bool:
    """Kiểm tra alias có phải cụm chung chung không."""
    return alias_lower in GENERIC_LOWER


def make_accent_variants(alias: str) -> list:
    """Trả về [alias, no_accent_variant] nếu có dấu."""
    variants = [alias]
    no_accent = unicodedata.normalize("NFD", alias)
    no_accent = "".join(c for c in no_accent if unicodedata.category(c) != "Mn")
    no_accent = unicodedata.normalize("NFC", no_accent)
    if no_accent != alias:
        variants.append(no_accent)
    return variants


def build_aliases_from_name(organ_name: str) -> list:
    """Sinh alias từ tên đầy đủ của công ty."""
    if not isinstance(organ_name, str) or not organ_name.strip():
        return []

    name = normalize_vi(organ_name)
    core = remove_legal_prefix(name)  # lowercase

    if not core or len(core) < 4:
        return []
    if len(core.split()) > 6:
        return []
    if is_generic(core):
        return []

    alias = title_case_vi(core)
    return make_accent_variants(alias)


def fetch_listing_df() -> pd.DataFrame:
    """
    Thử lần lượt các method của Listing() cho tới khi có DataFrame >100 dòng.
    In tên method và list columns.
    """
    from vnstock.api.listing import Listing

    listing = Listing()

    # method_name, call_kwargs
    methods_to_try = [
        ("all_symbols", {}),
        ("symbols_by_exchange", {"exchange": "ALL"}),
        ("symbols_by_exchange", {"exchange": "HOSE"}),
        ("symbols_by_industries", {}),
        ("symbols_by_group", {"group": "VN30"}),
    ]

    for method_name, kwargs in methods_to_try:
        if not hasattr(listing, method_name):
            print(f"[SKIP] Method '{method_name}' không tồn tại.")
            continue

        label = method_name + (f"({kwargs})" if kwargs else "()")
        print(f"\n[TRY] Listing().{label} ...")
        try:
            method = getattr(listing, method_name)
            df = method(**kwargs) if kwargs else method()

            if not isinstance(df, pd.DataFrame):
                print(f"  -> Không phải DataFrame (type={type(df)}), bỏ qua.")
                continue

            print(f"  OK  Rows={len(df)}, Columns={list(df.columns)}")

            if len(df) > 100:
                print(f"  => Dùng: {label}  ({len(df)} dòng)\n")
                return df
            else:
                print(f"  -> Chỉ {len(df)} dòng, thử method khác...")

        except Exception:
            import traceback
            print(f"  ERROR '{method_name}':")
            traceback.print_exc()

    raise RuntimeError(
        "Không có method nào trả về DataFrame >100 dòng.\n"
        "Kiểm tra kết nối mạng hoặc API vnstock."
    )


def detect_columns(df: pd.DataFrame):
    """Tự dò cột mã và cột tên. Trả về (ticker_col, name_col)."""
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
            f"Không tìm được cột mã trong DataFrame. "
            f"Columns: {list(df.columns)}"
        )

    print(f"[DETECT] ticker_col='{ticker_col}', name_col='{name_col}'")
    return ticker_col, name_col


def main() -> None:
    print("=" * 60)
    print("  BUILD TICKER DICT  (vnstock API)")
    print("=" * 60)

    # 1. Lấy dữ liệu -----------------------------------------------------------
    df = fetch_listing_df()

    # 2. Lưu raw ---------------------------------------------------------------
    raw_path = DATA_DIR / "listing_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] listing_raw.csv  ({len(df)} dòng) -> {raw_path}")

    # 3. Dò cột ----------------------------------------------------------------
    ticker_col, name_col = detect_columns(df)

    # 4. Lọc mã ^[A-Z]{3}$ ----------------------------------------------------
    tickers_raw = df[ticker_col].dropna().astype(str).str.strip().str.upper()
    valid_tickers = sorted({t for t in tickers_raw if re.match(r"^[A-Z]{3}$", t)})

    ticker_path = DATA_DIR / "tickers.txt"
    ticker_path.write_text("\n".join(valid_tickers) + "\n", encoding="utf-8")
    print(f"[SAVE] tickers.txt  ({len(valid_tickers)} mã) -> {ticker_path}")

    # 5. Xây dựng alias từ tên công ty -----------------------------------------
    alias_to_tickers: dict = {}   # alias_lower -> [ticker, ...]
    ticker_to_aliases: dict = {}  # ticker -> set[alias]

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

        # Loại alias trùng nhiều mã
        ambiguous = {a for a, ts in alias_to_tickers.items() if len(ts) > 1}
        print(f"[INFO] Aliases bị loại do trùng mã: {len(ambiguous)}")

        for ticker in ticker_to_aliases:
            ticker_to_aliases[ticker] = {
                a for a in ticker_to_aliases[ticker]
                if a.lower() not in ambiguous
            }

    # 6. Merge manual dict -----------------------------------------------------
    company_names: dict = {}

    for ticker in valid_tickers:
        aliases: set = set(ticker_to_aliases.get(ticker, set()))
        if ticker in MANUAL_DICT:
            for manual_alias in MANUAL_DICT[ticker]:
                for variant in make_accent_variants(manual_alias):
                    aliases.add(variant)
        if aliases:
            company_names[ticker] = sorted(aliases)

    # 7. Ghi JSON --------------------------------------------------------------
    json_path = DATA_DIR / "company_names.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(company_names, f, indent=1, ensure_ascii=False)
    print(f"[SAVE] company_names.json  ({len(company_names)} mã) -> {json_path}")

    # 8. Kết quả cuối ----------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  TICKERS : {len(valid_tickers):>5}  (PASS nếu >= 500)")
    print(f"  COMPANIES: {len(company_names):>5}  (PASS nếu >= 300)")
    print("=" * 60)

    if len(valid_tickers) >= 500:
        print("[PASS] tickers.txt OK")
    else:
        print(f"[FAIL] tickers.txt chỉ có {len(valid_tickers)} mã")

    if len(company_names) >= 300:
        print("[PASS] company_names.json OK")
    else:
        print(f"[FAIL] company_names.json chỉ có {len(company_names)} mã")


if __name__ == "__main__":
    main()