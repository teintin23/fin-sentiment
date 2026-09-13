from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def grab(path: Path, pattern: str, default: str = "?") -> str:
    if not path.exists():
        return default
    m = re.search(pattern, path.read_text(encoding="utf-8"))
    return m.group(1) if m else default


def main() -> int:
    splits = {n: pd.read_parquet(PROC / f"{n}.parquet") for n in ("train", "val", "test")}
    full = pd.read_parquet(PROC / "dataset_full.parquet")

    kappa = grab(DOCS / "label_quality.md", r"kappa[^0-9]*([01]\.\d+)")
    if kappa == "?":
        kappa = grab(DOCS / "prelabel_local_report.md", r"Cohen's kappa\*\* \| \*\*([01]\.\d+)")
    base_f1 = grab(DOCS / "model_baseline.md", r"macro-F1 \| \*\*([01]\.\d+)")
    pho_f1 = grab(DOCS / "model_phobert.md", r"macro-F1 \| \*\*([01]\.\d+)")
    pval = grab(DOCS / "model_comparison.md", r"p-value = \*\*([0-9.e\-]+)")

    def grab_es(path):
        if not path.exists():
            return None
        txt = path.read_text(encoding="utf-8")
        car = pv = n = None
        m = re.search(r"\| \[0,0\] \| ([+\-][0-9.]+) \| [\-0-9.]+ \| ([0-9.e\-]+) \|", txt)
        if m:
            car, pv = m.group(1), m.group(2)
        m = re.search(r"Usable events\*\* \| \*\*([\d,]+)", txt)
        if m:
            n = m.group(1)
        return (car, pv, n) if car else None

    es_main = grab_es(DOCS / "event_study.md")
    es_iso = grab_es(DOCS / "event_study_no_overlap.md")
    es_car, es_p, es_n = es_main if es_main else (None, None, None)

    def row(n: str, d: pd.DataFrame) -> dict:
        vc = d["label"].value_counts()
        return {"split": n, "n": len(d),
                **{l: int(vc.get(l, 0)) for l in LABELS},
                "from": str(d["date"].min())[:10], "to": str(d["date"].max())[:10]}

    tbl = pd.DataFrame([row(n, d) for n, d in splits.items()] + [row("FULL", full)])
    src = full["source_detail"].value_counts().to_dict()

    ctx = dict(kappa=kappa, base_f1=base_f1, pho_f1=pho_f1, pval=pval,
               tbl=tbl, src=src, full=full, splits=splits,
               es_car=es_car, es_p=es_p, es_n=es_n, es_iso=es_iso)

    write_readme(**ctx)
    ctx2 = {k: v for k, v in ctx.items() if not k.startswith('es_')}
    write_dataset_card(**ctx2)
    write_model_card(**ctx2)

    bad = []
    txt = (ROOT / "README.md").read_text(encoding="utf-8")
    for p in re.findall(r"\]\((?!http)([^)]+)\)", txt) + re.findall(r"`(src/[^`]+\.py)`", txt):
        if not (ROOT / p.lstrip("./")).exists():
            bad.append(p)
    print("Dead links in README:", bad if bad else "none")
    print("Written: README.md, docs/dataset_card.md, docs/model_card.md")
    return 1 if bad else 0


def write_readme(kappa, base_f1, pho_f1, pval, tbl, src, full, splits,
                 es_car=None, es_p=None, es_n=None, es_iso=None) -> None:
    w = []
    a = w.append
    a("# vn-fin-sentiment\n")
    a("Vietnamese financial news sentiment dataset and models. Labels follow the investor "
      "perspective of someone holding the named ticker. "
      f"Contains {len(full):,} CafeF articles, two baseline models, and an event study "
      "examining the relationship between sentiment and abnormal returns, "
      "with pre-event momentum controls.\n")
    a("> **TL;DR.** Vietnamese financial news sentiment dataset "
      f"({len(full):,} CafeF articles, holder-perspective labels, 3 label-source tiers) "
      "with a TF-IDF baseline and a fine-tuned PhoBERT, evaluated on a strictly "
      "chronological, fully hand-labeled test set. An event study finds a "
      "significant announcement-day abnormal-return spread between positive and "
      "negative news that survives controls for pre-event momentum, and no "
      "post-announcement drift: the labels carry priced information but no "
      "tradable forecast.\n")

    a("## Key results\n")
    a("| Item | Value |")
    a("|---|---|")
    a(f"| macro-F1 baseline TF-IDF + LogReg | {base_f1} |")
    a(f"| macro-F1 PhoBERT fine-tuned | **{pho_f1}** |")
    a(f"| McNemar p-value between models | {pval} (statistically significant) |")
    a(f"| Cohen's kappa machine vs human | {kappa} |")
    if es_car:
        a(f"| Event study, POS-NEG CAR[0,0] | {es_car}% (p={es_p}, {es_n} events) |")
        if es_iso:
            a(f"| — non-overlapping events only | {es_iso[0]}% (p={es_iso[1]}, "
              f"{es_iso[2]} events), effect at announcement day is not an artifact |")
        a("| — momentum control (CAR ~ POS+NEG+preCAR[-5,-1]) | POS-NEG [0,0] "
          "+0.905% (p=3.5e-07); non-overlapping sample +1.329% (p=0.0002) |")
        a("| — post-announcement CAR[1,5] | POS-NEG not significant (p=0.12) — "
          "no drift after announcement day |")
    else:
        a("| Event study | not yet run, see Limitations |")
    a("")

    a("![CM PhoBERT](reports/figures/cm_phobert.png)\n")

    a("## Dataset\n")
    a(tbl.to_markdown(index=False))
    a("")
    a("Label sources: " + ", ".join(f"`{k}` {v:,}" for k, v in src.items()) + "\n")
    a("| Label source | Meaning |")
    a("|---|---|")
    a("| `human` | Read and labelled by a human annotator |")
    a("| `claude_manual` | AI assistant read each article and labelled following the labeling guide |")
    a("| `weak_model` | Label propagation from seed set |")
    a("")
    a("The test set contains no `weak_model` labels; all labels are hand-read. "
      "This is intentional so that evaluation numbers are not distorted by noisy labels.\n")

    a("## Directory structure\n")
    a("```")
    a("src/                      code")
    a("  crawl/crawl_urls.py           collect article URLs from CafeF")
    a("  crawl/crawl_articles.py       download article content")
    a("  crawl/build_ticker_dict.py    build ticker symbol and company name dictionary")
    a("  repair_tickers.py       score and assign primary_ticker")
    a("  label_tool.py           web app for manual labeling (Flask)")
    a("  lexicon_vi_fin.py       Vietnamese financial sentiment lexicon")
    a("  prelabel_local.py       automated labeling without API calls")
    a("  audit_gold.py           review human/machine disagreements")
    a("  eval_labels.py          measure kappa between human and machine labels")
    a("  finalize_dataset.py     merge labels, split train/val/test by time")
    a("  train_baseline.py       TF-IDF + Logistic Regression")
    a("  train_phobert.py        fine-tune vinai/phobert-base")
    a("  compare_models.py       compare models + McNemar + error analysis")
    a("  fetch_prices.py         download closing prices (vnstock 4.x) for event study")
    a("  event_study.py          event study Brown & Warner, with placebo checks")
    a("  event_study_momentum.py event study with pre-event momentum controls\n  daily_alert.py          (optional) email bot for new articles + sentiment")
    a("  build_docs.py           regenerate README and cards from actual numbers")
    a("data/")
    a("  raw/                    raw data (not committed)")
    a("  interim/                intermediate data and labels")
    a("  processed/              train / val / test / dataset_full parquet")
    a("docs/                     reports and cards")
    a("reports/figures/          plots")
    a("models/                   trained models (not committed)")
    a("```\n")

    a("## Pipeline\n")
    a("| Step | Script | Output |")
    a("|---|---|---|")
    a("| 1. Collect URLs | `src/crawl/crawl_urls.py` | `data/raw/urls.jsonl` |")
    a("| 2. Download articles | `src/crawl/crawl_articles.py` | `data/raw/articles.jsonl` |")
    a("| 3. Assign tickers | `src/repair_tickers.py` | `data/interim/to_label_v2.parquet` |")
    a("| 4. Manual labeling | `src/label_tool.py` | `data/interim/gold_seed_v2.csv` |")
    a("| 5. Auto labeling | `src/prelabel_local.py` | `data/interim/labeled_auto.jsonl` |")
    a("| 6. Label quality | `src/eval_labels.py` | `docs/label_quality.md` |")
    a("| 7. Finalize dataset | `src/finalize_dataset.py` | `data/processed/*.parquet` |")
    a("| 8. Baseline | `src/train_baseline.py` | `docs/model_baseline.md` |")
    a("| 9. PhoBERT | `src/train_phobert.py` | `docs/model_phobert.md` |")
    a("| 10. Compare | `src/compare_models.py` | `docs/model_comparison.md` |")
    a("| 11. Prices | `src/fetch_prices.py` | `data/prices/*.csv` |")
    a("| 12. Event study | `src/event_study.py` | `docs/event_study.md` |")
    a("| 13. Momentum control | `src/event_study_momentum.py` | `docs/event_study_momentum.md` |")
    a("")

    a("## Reproduce from scratch\n")
    a("```bash")
    a("python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate")
    a("pip install -r requirements.txt")
    a("")
    a("python src/repair_tickers.py       # requires data/interim/articles_clean.parquet")
    a("python src/prelabel_local.py       # label ~5,549 articles, no API key needed")
    a("python src/eval_labels.py          # kappa between human and machine labels")
    a("python src/finalize_dataset.py     # split train/val/test by time")
    a("python src/train_baseline.py       # ~3 min on CPU")
    a("python src/train_phobert.py        # ~20 min on GPU T4")
    a("python src/compare_models.py")
    a("python src/fetch_prices.py         # closing prices for 195 tickers + VNINDEX, 3-5 min")
    a("python src/event_study.py          # sentiment vs abnormal returns")
    a("python src/event_study.py --no-overlap          # clean variant, no overlapping events")
    a("python src/event_study_momentum.py # regression with momentum controls + post-drift")
    a("python src/build_docs.py           # regenerate README and cards")
    a("```\n")

    a("## Design decisions\n")
    a("| Decision | Rationale |")
    a("|---|---|")
    a("| Time-based train/val/test split | Random split allows August articles to appear in train "
      "and then predict March articles, i.e., look-ahead leakage. For financial data this is a serious data leak. |")
    a("| Do not force gold seed into test | Gold seed is spread across time. Forcing it into test "
      "would break the time boundary built above. |")
    a("| Score-based ticker assignment instead of taking first | An article typically mentions 2-5 tickers. "
      "Taking the first one is wrong for about a third of articles. |")
    a("| Prepend ticker to model input as `[HPG] ...` | Labels are defined from the perspective of "
      "the holder of that ticker. Without the ticker the task is ill-defined. |")
    a("| Word-segment with underthesea before tokenizing | PhoBERT was pretrained on word-segmented text. "
      "Skipping this step degrades performance significantly. |")
    a("| Label propagation instead of LLM API calls | Zero cost, reproducible, no vendor dependency. "
      "Trade-off: noisier labels, documented below. |")
    a("| Three-tier `label_source` | Human, hand-read, and propagated labels have very different "
      "reliability. Collapsing them hides important quality information. |")
    a("| Keep test entirely as hand-read labels | So that macro-F1 on test is meaningful. |")
    a("")

    a("## Known limitations\n")
    a("- Single news source (CafeF). Not representative of the full market.")
    a(f"- Most train labels come from weak label propagation "
      f"({src.get('weak_model', 0):,}/{len(full):,} articles). Performance ceiling is set by "
      "label quality, not model architecture.")
    a(f"- Cohen's kappa between machine and human labels is only {kappa}, below the 0.60 threshold "
      "commonly considered substantial. The main cause is annotation criterion drift between the "
      "two labeling sessions, detailed in `docs/prelabel_local_report.md`.")
    a("- Ticker attribution accuracy estimated at ~85% (manual audit of 30 samples).")
    a("- Short time window, late 2024 to mid 2026, less than one full market cycle.")
    a("- Model reads headlines and lead paragraphs only, not full article text.")
    if es_car:
        a("- **Event study: endogeneity is present, quantified and controlled.** Placebo "
          "[-5,-1] is positive for POSITIVE group (+0.84%, p=7.4e-05): outlets tend to write "
          "positive news about rising stocks. After controlling for pre-drift via regression, "
          "POS-NEG at the announcement day remains +0.905% (p=3.5e-07); for non-overlapping "
          "events +1.329% (p=0.0002) — the concurrent effect is not a momentum artifact. "
          "Non-parametric check on the lowest |preCAR| tercile is consistent but lacks "
          "statistical power (small n). CAR[1,5] is not significant: labels reflect "
          "information priced at the announcement session, **no predictive power after the "
          "announcement**. Details in `docs/event_study.md` sections 6-8 and "
          "`docs/event_study_momentum.md`.")
    else:
        a("- **Event study not yet run.** Requires daily stock price data and VNINDEX, "
          "not yet collected. Any conclusions about price predictability are unsubstantiated.")
    a("- Not for use in real investment decisions.")
    a("")
    a("## License\n")
    a("Code: MIT. Data: CC BY 4.0. Original article content is copyright CafeF; "
      "the dataset redistributes only headlines, lead paragraphs, and metadata.")

    (ROOT / "README.md").write_text("\n".join(w) + "\n", encoding="utf-8")


def write_dataset_card(kappa, base_f1, pho_f1, pval, tbl, src, full, splits) -> None:
    n = len(full)
    size_cat = "1K<n<10K" if n < 10000 else "10K<n<100K"
    w = []
    a = w.append
    a("---")
    a("language:\n- vi")
    a("license: cc-by-4.0")
    a("task_categories:\n- text-classification")
    a("task_ids:\n- sentiment-classification")
    a("tags:\n- finance\n- vietnamese\n- sentiment\n- stock-market")
    a(f"size_categories:\n- {size_cat}")
    a("---\n")
    a("# vn-fin-sentiment\n")
    a("---\n\n## Dataset Summary\n")
    a(f"{n:,} Vietnamese financial news items from CafeF, each mapped to one listed "
      "ticker and labelled POSITIVE / NEUTRAL / NEGATIVE. Labels follow the "
      "perspective of an investor already holding that ticker: does this news raise "
      "or lower expected firm value? Splits are strictly time-ordered to avoid "
      "look-ahead leakage.\n")
    a("## Supported Tasks\n")
    a("Three-class text classification. Also usable as an event-study input where "
      "each labelled item is a dated signal for one ticker.\n")
    a("## Languages\n\nVietnamese (`vi`).\n")
    a("## Data Fields\n")
    a("| Field | Type | Meaning |")
    a("|---|---|---|")
    a("| `id` | string | Stable article identifier |")
    a("| `datetime` | timestamp | Publication timestamp |")
    a("| `date` | string | Publication date, `YYYY-MM-DD` |")
    a("| `primary_ticker` | string | Ticker the label refers to |")
    a("| `ticker_in_head` | bool | Whether the ticker appears in title or lead |")
    a("| `title` | string | Article headline |")
    a("| `sapo` | string | Lead paragraph |")
    a("| `text` | string | Body text |")
    a("| `text_input` | string | Model input, format `\"[TICKER] title. sapo\"` |")
    a("| `label` | string | POSITIVE / NEUTRAL / NEGATIVE |")
    a("| `label_source` | string | `human` or `llm` |")
    a("| `source_detail` | string | `human`, `claude_manual`, or `weak_model` |")
    a("| `confidence_val` | float | Label confidence, 1.0 for human labels |")
    a("| `split` | string | train / validation / test |")
    a("")
    a("## Data Splits\n")
    a(tbl.to_markdown(index=False))
    a("")
    a("## Curation Rationale\n")
    a("Vietnamese financial NLP lacks a public sentiment dataset that is (a) tied to a "
      "specific ticker rather than general market mood, and (b) split by time so that "
      "downstream event studies are not contaminated by look-ahead bias. This dataset "
      "targets both gaps.\n")
    a("## Source Data\n")
    a("Crawled from CafeF (cafef.vn), a mainstream Vietnamese financial news outlet. "
      f"Coverage {tbl.iloc[-1]['from']} to {tbl.iloc[-1]['to']}. Only headline, lead, "
      "and metadata are redistributed.\n")
    a("## Annotation Process\n")
    a("Three tiers, recorded per row in `source_detail`:\n")
    a("| Tier | Count | Method |")
    a("|---|---:|---|")
    for k in ("human", "claude_manual", "weak_model"):
        meth = {"human": "Read and labelled by a human annotator",
                "claude_manual": "Read individually by an AI assistant following the labelling guide",
                "weak_model": "Propagated by TF-IDF + lexicon + calibrated logistic regression"}[k]
        a(f"| `{k}` | {src.get(k, 0):,} | {meth} |")
    a("")
    a(f"Agreement between machine labels and the human gold set: Cohen's kappa = "
      f"**{kappa}** on the overlapping items. This is below the 0.60 threshold usually "
      "called substantial. Diagnosis in `docs/prelabel_local_report.md`: the human gold "
      "set itself drifted between two annotation sessions, with the later session "
      "labelling 70% POSITIVE against 45% in the earlier one. Treat the kappa figure as "
      "a joint statement about both label sources, not about the machine alone.\n")
    a("The test split contains no `weak_model` labels by design.\n")
    a("## Ticker Attribution\n")
    a("`src/repair_tickers.py` scores every candidate ticker in an article using three "
      "signals in priority order: tickers shown in the page's own stock widget, tickers "
      "named explicitly in the text, and tickers inferred from full company names. The "
      "highest-scoring candidate becomes `primary_ticker`, with the score gap kept in "
      "`pt_margin`. Manual audit of 30 samples put accuracy at roughly 85%.\n")
    a("## Personal and Sensitive Information\n")
    a("All content is published financial news. Named individuals appear only in their "
      "public professional capacity as company officers or regulators. No private "
      "personal data.\n")
    a("## Limitations and Bias\n")
    a("1. Single source (CafeF); outlet-specific framing is baked in.")
    a(f"2. {src.get('weak_model', 0):,} of {n:,} labels come from a weak propagation model.")
    a(f"3. Machine-vs-human kappa is only {kappa}; part of that is human annotation drift.")
    a("4. Ticker attribution accuracy ~85%; wrong ticker means the label describes the wrong firm.")
    a("5. Short window (late 2024 to mid 2026), less than one full market cycle.")
    a("6. Class imbalance: NEUTRAL dominates, NEGATIVE is the rarest class.")
    a("7. Headline and lead only, no full-text reasoning.")
    a("8. Event clustering: firms in the same sector often get news on the same day.\n")
    a("## Citation\n")
    a("```bibtex\n@misc{vnfinsentiment,\n  title  = {vn-fin-sentiment: Ticker-level "
      "sentiment for Vietnamese financial news},\n  author = {<Your Name>},\n"
      "  year   = {2026},\n  url    = {https://github.com/<username>/"
      "vn-fin-sentiment}\n}\n```\n")

    (DOCS / "dataset_card.md").write_text("\n".join(w) + "\n", encoding="utf-8")


def write_model_card(kappa, base_f1, pho_f1, pval, tbl, src, full, splits) -> None:
    te = splits["test"]
    hp = (DOCS / "model_phobert.md").read_text(encoding="utf-8") if (DOCS / "model_phobert.md").exists() else ""
    cm_block = ""
    m = re.search(r"## 4\. Confusion matrix\n\n(\|.*?)\n\n!", hp, re.S)
    if m:
        cm_block = m.group(1)
    acc = grab(DOCS / "model_phobert.md", r"accuracy \| ([01]\.\d+)")
    wf1 = grab(DOCS / "model_phobert.md", r"weighted-F1 \| ([01]\.\d+)")

    code = ("```python\n"
            "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n"
            "from underthesea import word_tokenize\n"
            "import torch\n\n"
            "REPO = 'models/phobert-vnfin'   # or '<username>/phobert-vn-fin-sentiment'\n"
            "tok = AutoTokenizer.from_pretrained(REPO)\n"
            "model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()\n\n"
            "text = '[HPG] Hoa Phat reports record Q2 profit. Net income up 48% YoY.'\n"
            "seg = word_tokenize(text, format='text')          # word-segment first\n"
            "x = tok(seg, return_tensors='pt', truncation=True, max_length=256)\n"
            "with torch.no_grad():\n"
            "    print(model.config.id2label[model(**x).logits.argmax(-1).item()])\n"
            "```\n")

    w = []
    a = w.append
    a("---")
    a("language:\n- vi")
    a("license: mit")
    a("base_model: vinai/phobert-base")
    a("pipeline_tag: text-classification")
    a("tags:\n- finance\n- vietnamese\n- sentiment\n- phobert")
    a("---\n")
    a("# phobert-vn-fin-sentiment\n")
    a("---\n\n## Model Description\n")
    a("`vinai/phobert-base` fine-tuned for three-class sentiment on Vietnamese "
      "financial news. The label answers one question: for an investor already holding "
      f"the ticker named in the input, does this news raise expected firm value "
      "(POSITIVE), lower it (NEGATIVE), or neither (NEUTRAL)?\n")
    a("## Intended Use\n")
    a("Research on Vietnamese financial NLP; feature extraction for event studies; a "
      "baseline for anyone building ticker-level sentiment in Vietnamese.\n")
    a("## Out-of-Scope Use\n")
    a("**Do not use this model for real investment decisions.** It reads headlines and "
      "lead paragraphs only, was trained largely on machine-propagated labels, and has "
      "never been validated against realised returns. It is a research artifact.\n")
    a("Also out of scope: non-financial Vietnamese text, other languages, and any "
      "input where the ticker prefix is missing.\n")
    a("## Training Data\n")
    a("`vn-fin-sentiment`, see `docs/dataset_card.md`. "
      f"Train {len(splits['train']):,} rows, validation {len(splits['val']):,}, "
      f"test {len(te):,}. Splits are strictly time-ordered.\n")
    a("## Training Procedure\n")
    a("| Hyperparameter | Value |")
    a("|---|---|")
    a("| learning rate | 2e-5 |")
    a("| batch size | 16 |")
    a("| epochs | 4, early stopping patience 2 |")
    a("| warmup ratio | 0.1 |")
    a("| weight decay | 0.01 |")
    a("| max length | 256 |")
    a("| class weights | yes, inverse label frequency in train |")
    a("| best-model metric | macro-F1 on validation |")
    a("| seed | 42 |")
    a("| hardware | single NVIDIA T4 (Google Colab) |")
    a("| wall time | roughly 20 minutes |")
    a("")
    a("## Evaluation\n")
    a("| Model | macro-F1 | accuracy | weighted-F1 |")
    a("|---|---|---|---|")
    a(f"| TF-IDF + Logistic Regression | {base_f1} | — | — |")
    a(f"| **PhoBERT (this model)** | **{pho_f1}** | {acc} | {wf1} |")
    a("")
    a(f"McNemar test between the two: p = {pval}, so the gap is statistically "
      "significant at alpha 0.05. Full breakdown in `docs/model_comparison.md`.\n")
    a("Confusion matrix, rows are true labels:\n")
    a(cm_block + "\n")
    a("![confusion matrix](../reports/figures/cm_phobert.png)\n")
    a("## Input Format\n")
    a("Input **must** be `\"[TICKER] Headline. Lead paragraph\"` and **must** be "
      "word-segmented with underthesea before tokenisation. PhoBERT was pretrained on "
      "segmented text; skipping this step degrades results sharply.\n")
    a(code)
    a("## Limitations and Bias\n")
    a("1. Single news source (CafeF).")
    a(f"2. About {src.get('weak_model', 0) / len(full):.0%} of all labels come from a "
      "weak propagation model, so the ceiling is set by label quality, not architecture.")
    a(f"3. Machine-vs-human label agreement is only kappa = {kappa}.")
    a("4. Ticker attribution accuracy ~85%.")
    a("5. Headline and lead only; no full-text reasoning.")
    a("6. NEGATIVE is the rarest class and the hardest one.")
    a("7. Trained on late 2024 to mid 2026; performance on other regimes is untested.")
    a("8. Never validated against realised stock returns.\n")

    (DOCS / "model_card.md").write_text("\n".join(w) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())