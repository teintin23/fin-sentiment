from __future__ import annotations

import argparse
import io
import json
import os
import re
import smtplib
import sys
import time
from datetime import datetime
from email.message import EmailMessage
from html import escape
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src" / "crawl"))
import crawl_articles as ca  # noqa: E402

CONFIG = ROOT / "config" / "alert.json"
WATCHLIST = ROOT / "config" / "watchlist.txt"
STATE = ROOT / "data" / "alert_state.json"
MODEL_DIR = ROOT / "models" / "phobert-vnfin"
PREVIEW = ROOT / "reports" / "alert_preview.html"

ZONES = {"stock": 18831, "banking": 18834, "real_estate": 18835,
         "corporate": 18836}
ARTICLE_PATTERN = re.compile(r"\d{6,}\.chn")
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

CONFIG_TEMPLATE = {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 465,
    "smtp_user": "you@gmail.com",
    "smtp_password": "app-password-16-chars",
    "email_to": "you@gmail.com",
    "min_confidence": 0.70,
    "pages_per_zone": 3,
    "max_articles_per_run": 120,
}


def list_new_urls(pages: int, seen: set[str]) -> list[str]:
    urls: list[str] = []
    for zone, zid in ZONES.items():
        for page in range(1, pages + 1):
            u = f"https://cafef.vn/timelinelist/{zid}/{page}.chn"
            try:
                r = requests.get(u, headers=ca.HEADERS, timeout=20)
                r.encoding = "utf-8"
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR listing {zone} p{page}: {e}")
                continue
            for m in re.finditer(r'href="([^"]+)"', r.text):
                href = m.group(1)
                if ARTICLE_PATTERN.search(href):
                    full = href if href.startswith("http") else "https://cafef.vn" + href
                    if full not in seen:
                        urls.append(full)
            time.sleep(0.4)
    return list(dict.fromkeys(urls))


def fetch_articles(urls: list[str], whitelist: set[str]) -> list[dict]:
    out = []
    for u in urls:
        try:
            r = requests.get(u, headers=ca.HEADERS, timeout=20)
            r.encoding = "utf-8"
            rec = ca.parse_article(r.text, u, whitelist)
        except Exception:  # noqa: BLE001
            continue
        if rec["title"]:
            out.append(rec)
        time.sleep(0.5)
    return out


def pick_ticker(rec: dict) -> str | None:
    for tier in ("tickers_widget", "tickers_explicit", "tickers_by_name"):
        cands = rec.get(tier) or []
        if cands:
            in_title = [c for c in cands if c in rec["title"].upper()]
            return (in_title or cands)[0]
    return None


class Sentiment:
    def __init__(self) -> None:
        if not (MODEL_DIR / "config.json").exists():
            sys.exit(
                f"Model not found at {MODEL_DIR}.\n"
                "Copy the trained model directory (config.json, model.safetensors, "
                "tokenizer...) there.")
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        from underthesea import word_tokenize
        self.torch = torch
        self.seg = word_tokenize
        self.tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def predict(self, texts: list[str], batch: int = 16) -> list[tuple[str, float]]:
        res = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                segd = [self.seg(t, format="text") for t in texts[i:i + batch]]
                x = self.tok(segd, return_tensors="pt", truncation=True,
                             max_length=256, padding=True)
                probs = self.torch.softmax(self.model(**x).logits, dim=-1)
                for row in probs:
                    j = int(row.argmax())
                    res.append((self.id2label[j], float(row[j])))
        return res


def build_email(items: list[dict], watch: set[str], min_conf: float
                ) -> tuple[str, str, str]:
    wl_neg = [x for x in items if x["ticker"] in watch and x["label"] == "NEGATIVE"]
    wl_rest = [x for x in items if x["ticker"] in watch and x["label"] != "NEGATIVE"]
    mkt_neg = [x for x in items if x["ticker"] not in watch
               and x["label"] == "NEGATIVE" and x["conf"] >= min_conf]
    mkt_pos = [x for x in items if x["ticker"] not in watch
               and x["label"] == "POSITIVE" and x["conf"] >= min_conf]

    today = datetime.now().strftime("%d/%m/%Y")
    n_lab = {l: sum(x["label"] == l for x in items) for l in LABELS}
    subject = (f"[vnfin] {today}: "
               + (f"{len(wl_neg)} NEGATIVE watchlist, " if wl_neg else "")
               + f"{n_lab['NEGATIVE']} NEG / {n_lab['POSITIVE']} POS "
                 f"/ {len(items)} articles")

    def li_html(x: dict) -> str:
        col = {"NEGATIVE": "#c0392b", "POSITIVE": "#1e8449",
               "NEUTRAL": "#7f8c8d"}[x["label"]]
        return (f'<li style="margin:6px 0">'
                f'<b>{escape(x["ticker"])}</b> '
                f'<span style="color:{col}">{x["label"]} {x["conf"]:.0%}</span> — '
                f'<a href="{escape(x["url"])}">{escape(x["title"])}</a></li>')

    def li_text(x: dict) -> str:
        return f'  {x["ticker"]:5s} {x["label"]:8s} {x["conf"]:.0%}  {x["title"]}\n    {x["url"]}'

    def section(title: str, xs: list[dict], note: str = "") -> tuple[str, str]:
        if not xs:
            return "", ""
        h = f"<h3 style='margin:16px 0 4px'>{title}</h3>"
        if note:
            h += f"<p style='color:#666;font-size:13px;margin:2px 0'>{note}</p>"
        h += "<ul style='padding-left:18px'>" + "".join(li_html(x) for x in xs) + "</ul>"
        t = f"\n== {title} ==\n" + (note + "\n" if note else "") \
            + "\n".join(li_text(x) for x in xs) + "\n"
        return h, t

    drift = ("Note from event study: NEGATIVE articles have average CAR of "
             "-0.45% on day 0, drifting to -1.42% over 5 sessions. "
             "Bad news is often not priced in advance. Consider reducing position "
             "early rather than holding. Not investment advice.")

    parts = [
        section("WARNING — NEGATIVE NEWS ON WATCHLIST", wl_neg, drift),
        section("Watchlist — other news", wl_rest),
        section(f"Negative market-wide (conf >= {min_conf:.0%})", mkt_neg),
        section(f"Positive market-wide (conf >= {min_conf:.0%})", mkt_pos),
    ]
    html = ("<div style='font-family:Segoe UI,Arial,sans-serif;font-size:14px'>"
            f"<p>{len(items)} new articles with identifiable ticker. "
            f"NEG {n_lab['NEGATIVE']} / NEU {n_lab['NEUTRAL']} / POS {n_lab['POSITIVE']}.</p>"
            + "".join(p[0] for p in parts)
            + "<p style='color:#999;font-size:12px'>Automated bot from repo "
              "vn-fin-sentiment. Model F1 0.79, ticker accuracy ~85% — read "
              "original article before acting.</p></div>")
    text = (f"{len(items)} new articles. NEG {n_lab['NEGATIVE']} / NEU "
            f"{n_lab['NEUTRAL']} / POS {n_lab['POSITIVE']}.\n"
            + "".join(p[1] for p in parts))
    return subject, text, html


def send_email(cfg: dict, subject: str, text: str, html: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["smtp_user"]
    msg["To"] = cfg["email_to"]
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    port = int(cfg["smtp_port"])
    if port == 465:
        with smtplib.SMTP_SSL(cfg["smtp_host"], port, timeout=30) as s:
            s.login(cfg["smtp_user"], cfg["smtp_password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(cfg["smtp_host"], port, timeout=30) as s:
            try:
                s.starttls()
                s.login(cfg["smtp_user"], cfg["smtp_password"])
            except smtplib.SMTPNotSupportedError:
                pass
            s.send_message(msg)


def load_json(p: Path, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def load_watchlist() -> set[str]:
    if not WATCHLIST.exists():
        return set()
    raw = WATCHLIST.read_text(encoding="utf-8")
    return {t.strip().upper() for t in re.split(r"[\s,;#]+", raw)
            if re.fullmatch(r"[A-Z]{3}[0-9]?", t.strip().upper())}


def run(args) -> int:
    cfg = load_json(CONFIG, None)
    if cfg is None:
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(CONFIG_TEMPLATE, ensure_ascii=False,
                                     indent=2), encoding="utf-8")
        if not WATCHLIST.exists():
            WATCHLIST.write_text("# one ticker per line, e.g.:\nHPG\nFPT\n",
                                 encoding="utf-8")
        print(f"Created {CONFIG} and {WATCHLIST}. Fill in credentials then re-run.\n"
              "For Gmail use an App Password, not your regular password.")
        return 1

    if args.test_email:
        send_email(cfg, "[vnfin] test email",
                   "SMTP working.", "<b>SMTP working.</b>")
        print(f"Test email sent to {cfg['email_to']}")
        return 0

    watch = load_watchlist()
    state = load_json(STATE, {"seen": []})
    seen = set(state["seen"])
    print(f"Watchlist: {sorted(watch) or 'EMPTY'} | known articles: {len(seen)}")

    urls = list_new_urls(int(cfg.get("pages_per_zone", 3)), seen)
    urls = urls[: int(cfg.get("max_articles_per_run", 120))]
    print(f"New articles on listing: {len(urls)}")
    if not urls:
        print("No new articles, skipping.")
        return 0

    whitelist = ca.load_ticker_whitelist()
    arts = fetch_articles(urls, whitelist)
    for a_ in arts:
        a_["ticker"] = pick_ticker(a_)
    items = [a_ for a_ in arts if a_["ticker"]]
    print(f"Fetched {len(arts)} articles, {len(items)} with ticker")

    if items:
        model = Sentiment()
        texts = [f"[{x['ticker']}] {x['title']}. {x['sapo']}" for x in items]
        for x, (lab, conf) in zip(items, model.predict(texts)):
            x["label"], x["conf"] = lab, conf

        subject, text, html = build_email(items, watch,
                                          float(cfg.get("min_confidence", 0.7)))
        PREVIEW.parent.mkdir(parents=True, exist_ok=True)
        PREVIEW.write_text(html, encoding="utf-8")
        if args.dry_run:
            print(f"[dry-run] not sent. Preview: {PREVIEW}\n{subject}")
            return 0
        send_email(cfg, subject, text, html)
        print(f"Sent: {subject}")
    elif args.dry_run:
        print("[dry-run] no articles with ticker.")
        return 0

    seen |= {a_["url"] for a_ in arts} | set(urls)
    state["seen"] = sorted(seen)[-20000:]
    state["last_run"] = datetime.now().isoformat(timespec="seconds")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state), encoding="utf-8")
    return 0


def selftest() -> int:
    import pandas as pd
    df = pd.read_parquet(ROOT / "data/processed/dataset_full.parquet")
    take = pd.concat([df[df["label"] == l].head(4) for l in LABELS])
    items = [{"ticker": r["primary_ticker"], "title": r["title"],
              "sapo": r["sapo"], "url": r["url"], "label": r["label"],
              "conf": 0.88} for _, r in take.iterrows()]
    watch = {items[0]["ticker"], items[-1]["ticker"]}
    subject, text, html = build_email(items, watch, 0.7)
    out = ROOT / "reports" / "alert_selftest.html"
    out.write_text(html, encoding="utf-8")
    expected = [x for x in items
                if x["ticker"] in watch
                or (x["label"] != "NEUTRAL" and x["conf"] >= 0.7)]
    ok = all(x["title"][:30] in text or escape(x["title"][:30]) in html
             for x in expected) and len(expected) >= 8
    print(f"build_email: {'OK' if ok else 'FAIL'} — see {out}\n  {subject}")

    rec = {"tickers_widget": [], "tickers_explicit": ["FPT", "HPG"],
           "tickers_by_name": ["VNM"], "title": "HPG reports record profit"}
    ok2 = pick_ticker(rec) == "HPG"
    print(f"pick_ticker title priority: {'OK' if ok2 else 'FAIL'}")

    ok3 = True
    try:
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, "-m", "aiosmtpd", "-n", "-l", "localhost:8025"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        send_email({"smtp_host": "localhost", "smtp_port": 8025,
                    "smtp_user": "t@local", "smtp_password": "x",
                    "email_to": "t@local"}, subject, text, html)
        proc.terminate()
        print("SMTP (local sink): OK")
    except Exception as e:  # noqa: BLE001
        ok3 = False
        print(f"SMTP (local sink): SKIP ({e}) — install aiosmtpd to test")
    print("PASS" if ok and ok2 else "FAIL")
    return 0 if ok and ok2 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="create config template only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-email", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.init and not CONFIG.exists():
        args.test_email = False
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
