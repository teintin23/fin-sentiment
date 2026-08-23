"""Collect article URLs from CafeF timeline endpoints."""
import json
import random
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

BASE_URL = "https://cafef.vn"
ARTICLE_PATTERN = re.compile(r"\d{6,}\.chn")

# Verified CafeF zone ids. Add more after probe_zones.py confirms them.
ZONES = {
    "stock":       18831,
    "banking":     18834,
    "real_estate": 18835,
    "corporate":   18836,
}

OUTPUT_PATH = Path("data/raw/urls.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def fetch_page_links(zone_id, page):
    """Return set of absolute article URLs on one timeline page."""
    url = f"{BASE_URL}/timelinelist/{zone_id}/{page}.chn"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "lxml")
    links = set()
    for a in soup.select("a[href]"):
        href = a["href"]
        if ARTICLE_PATTERN.search(href):
            links.add(href if href.startswith("http") else BASE_URL + href)
    return links


def crawl_zone(zone_name, zone_id, max_page, seen, out_file):
    """Walk one zone until pages stop returning articles."""
    dead_pages = 0
    added = 0

    for page in range(1, max_page + 1):
        try:
            links = fetch_page_links(zone_id, page)
        except Exception as exc:
            print(f"  [ERROR] {zone_name} p{page}: {exc}")
            time.sleep(5)
            continue

        # Stop condition depends on whether the PAGE is empty,
        # not on whether the URLs are new. Re-runs must not stop early.
        if not links:
            dead_pages += 1
            if dead_pages >= 3:
                print(f"  {zone_name}: 3 empty pages at p{page}, stopping.")
                break
            continue
        dead_pages = 0

        new_links = links - seen
        for link in new_links:
            record = {"url": link, "zone": zone_name}
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        seen |= new_links
        added += len(new_links)

        if page % 20 == 0:
            print(f"  {zone_name} p{page}: +{added} this zone | {len(seen)} total")

        time.sleep(random.uniform(0.6, 1.2))

    print(f"  {zone_name}: done, +{added} new URLs")


def main(max_page=400):
    seen = set()
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8") as f:
            seen = {json.loads(line)["url"] for line in f}
        print(f"Resuming with {len(seen)} URLs already collected")

    with OUTPUT_PATH.open("a", encoding="utf-8") as out_file:
        for zone_name, zone_id in ZONES.items():
            print(f"\n=== zone {zone_name} (id {zone_id}) ===")
            crawl_zone(zone_name, zone_id, max_page, seen, out_file)

    print(f"\nTOTAL: {len(seen)} URLs")


if __name__ == "__main__":
    main(max_page=600)   # smoke test first, then raise to 400