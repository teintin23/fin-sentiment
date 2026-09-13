import hashlib
import json
import random
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

INPUT_PATH = Path("data/raw/urls.jsonl")
OUTPUT_PATH = Path("data/raw/articles.jsonl")
TICKER_PATH = Path("data/tickers.txt")
COMPANY_NAMES_PATH = Path("data/company_names.json")

URL_TIMESTAMP = re.compile(r"(\d{3})(\d{6})(\d{6})\d*\.chn$")

TICKER_WIDGET_SELECTOR = "div.chisochungkhoan h2.title_box a"

WIDGET_TEXT = re.compile(
    r"([A-Z]{3}[0-9]?)\s*:\s*Giá hiện tại\s*Thay đổi\s*Xem hồ sơ doanh nghiệp\s*TIN MỚI\s*"
)

BLACKLIST = {"USD", "GDP", "CEO", "FDI", "HSX", "ETF", "IPO", "EPS", "ROE",
             "ROA", "CPI", "FED", "ECB", "COO", "CFO", "CTO", "TPP", "WTO",
             "VAT", "BOT", "PPP", "GMT", "USA", "PMI", "NIM", "CAR", "ESG",
             "SME", "CCP", "VKS", "TOP", "MTV", "KKT", "KCN", "SJC", "ATM",
             "POS", "QR",  "AI",  "API"}

JUNK_SELECTORS = [
    "script", "style", "table.picture", ".VCSortableInPreviewMode",
    ".admwrapper", ".box-tinlienquan", ".relate-news", ".contentbottom",
    "div.chisochungkhoan",
]


def load_ticker_whitelist():
    if not TICKER_PATH.exists():
        return set()
    raw = TICKER_PATH.read_text(encoding="utf-8")
    return {t.strip().upper() for t in re.split(r"[\s,;]+", raw) if t.strip()}


def load_company_names():
    if not COMPANY_NAMES_PATH.exists():
        return []
    data = json.loads(COMPANY_NAMES_PATH.read_text(encoding="utf-8"))
    pairs = [(alias, code) for code, aliases in data.items() for alias in aliases]
    return sorted(pairs, key=lambda p: -len(p[0]))


COMPANY_NAME_PAIRS = load_company_names()


def parse_url_timestamp(url):
    m = URL_TIMESTAMP.search(url)
    if not m:
        return None, None
    _, ymd, hms = m.groups()
    return (f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}",
            f"{hms[:2]}:{hms[2:4]}:{hms[4:6]}")


def extract_tickers(soup, title, sapo, raw_body, whitelist):
    widget = []
    for a in soup.select(TICKER_WIDGET_SELECTOR):
        code = a.get_text(strip=True).replace(":", "").strip().upper()
        if re.fullmatch(r"[A-Z]{3}[0-9]?", code):
            widget.append(code)

    head = f"{title} {sapo}"
    head_clean = re.sub(
        r"TP\.?\s*HCM\b|Tp\.?\s*HCM\b|TP\.?\s*H\u1ed3\s*Ch\u00ed\s*Minh",
        " ", head, flags=re.IGNORECASE)
    head_clean = re.sub(
        r"\b(?:v\u00e0ng\s+SJC|SJC\s+v\u00e0ng)\b",
        " ", head_clean, flags=re.IGNORECASE)
    explicit = [c for c in re.findall(r"\b([A-Z]{3})\b", head_clean)
                if c not in BLACKLIST and (not whitelist or c in whitelist)]

    scan = f"{title} {sapo} {raw_body[:1500]}"
    by_name = [code for alias, code in COMPANY_NAME_PAIRS if alias in scan]

    return sorted(set(widget)), sorted(set(explicit)), sorted(set(by_name))


def clean_body(soup):
    content = soup.select_one("div.detail-content")
    if content is None:
        return ""
    for sel in JUNK_SELECTORS:
        for el in content.select(sel):
            el.decompose()
    text = WIDGET_TEXT.sub("", content.get_text(" ", strip=True))
    return re.sub(r"\s+", " ", text).strip()


def parse_article(html, url, whitelist):
    soup = BeautifulSoup(html, "lxml")

    def text_of(selector):
        el = soup.select_one(selector)
        return el.get_text(" ", strip=True) if el else ""

    title = text_of("h1.title")
    sapo = text_of(".sapo")

    content_el = soup.select_one("div.detail-content")
    raw_body = content_el.get_text(" ", strip=True) if content_el else ""

    widget_t, explicit_t, name_t = extract_tickers(soup, title, sapo, raw_body, whitelist)
    body = clean_body(soup)
    date, clock = parse_url_timestamp(url)

    return {
        "id": hashlib.md5(url.encode()).hexdigest()[:12],
        "url": url,
        "title": title,
        "sapo": sapo,
        "date": date,
        "time": clock,
        "pdate_raw": text_of(".pdate"),
        "tickers_widget": widget_t,
        "tickers_explicit": explicit_t,
        "tickers_by_name": name_t,
        "tickers_any": sorted(set(widget_t) | set(explicit_t) | set(name_t)),
        "body": body[:5000],
        "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main(limit=None):
    whitelist = load_ticker_whitelist()
    print(f"Whitelist    : {len(whitelist)} tickers")
    print(f"Company alias: {len(COMPANY_NAME_PAIRS)} entries"
          f"{'  <-- EMPTY, create data/company_names.json' if not COMPANY_NAME_PAIRS else ''}")

    with INPUT_PATH.open(encoding="utf-8") as f:
        urls = list(dict.fromkeys(json.loads(line)["url"] for line in f))

    done = set()
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8") as f:
            done = {json.loads(line)["url"] for line in f}

    todo = [u for u in urls if u not in done]
    random.shuffle(todo)
    if limit:
        todo = todo[:limit]
    print(f"{len(urls)} total | {len(done)} done | {len(todo)} to fetch\n")

    ok = no_title = failed = 0
    hit_widget = hit_explicit = hit_name = hit_any = 0

    with OUTPUT_PATH.open("a", encoding="utf-8") as out:
        for url in tqdm(todo):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=20)
                resp.encoding = "utf-8"
                record = parse_article(resp.text, url, whitelist)
            except Exception:
                failed += 1
                time.sleep(2)
                continue

            if not record["title"]:
                no_title += 1
                continue

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            ok += 1
            hit_widget += bool(record["tickers_widget"])
            hit_explicit += bool(record["tickers_explicit"])
            hit_name += bool(record["tickers_by_name"])
            hit_any += bool(record["tickers_any"])

            time.sleep(random.uniform(0.5, 1.0))

    print(f"\nparsed         : {ok}")
    if ok:
        print(f"  widget       : {hit_widget:>5} ({hit_widget / ok * 100:5.1f}%)")
        print(f"  explicit code: {hit_explicit:>5} ({hit_explicit / ok * 100:5.1f}%)")
        print(f"  company name : {hit_name:>5} ({hit_name / ok * 100:5.1f}%)")
        print(f"  ANY signal   : {hit_any:>5} ({hit_any / ok * 100:5.1f}%)")
    print(f"empty title    : {no_title}")
    print(f"request failed : {failed}")


if __name__ == "__main__":
    main(limit=20000)