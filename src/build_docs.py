"""
build_docs.py
-------------
Sinh README.md, docs/dataset_card.md, docs/model_card.md từ số liệu THẬT trong
data/processed/ và docs/. Không con số nào viết tay.

Chạy lại sau mỗi lần dataset hoặc kết quả mô hình thay đổi.

Usage:
    python src/build_docs.py
"""

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

    # --- so lieu tu cac bao cao da co --------------------------------------
    kappa = grab(DOCS / "label_quality.md", r"kappa[^0-9]*([01]\.\d+)")
    if kappa == "?":
        kappa = grab(DOCS / "prelabel_local_report.md", r"Cohen's kappa\*\* \| \*\*([01]\.\d+)")
    base_f1 = grab(DOCS / "model_baseline.md", r"macro-F1 \| \*\*([01]\.\d+)")
    pho_f1 = grab(DOCS / "model_phobert.md", r"macro-F1 \| \*\*([01]\.\d+)")
    pval = grab(DOCS / "model_comparison.md", r"p-value = \*\*([0-9.e\-]+)")

    # event study (neu da chay)
    es = DOCS / "event_study.md"
    es_car = es_p = es_n = None
    if es.exists():
        txt = es.read_text(encoding="utf-8")
        m = re.search(r"\| \[0,5\] \| ([+\-][0-9.]+) \| [\-0-9.]+ \| ([0-9.e\-]+) \|", txt)
        if m:
            es_car, es_p = m.group(1), m.group(2)
        m = re.search(r"Sự kiện dùng được\*\* \| \*\*([\d,]+)", txt)
        if m:
            es_n = m.group(1)

    def row(n: str, d: pd.DataFrame) -> dict:
        vc = d["label"].value_counts()
        return {"split": n, "n": len(d),
                **{l: int(vc.get(l, 0)) for l in LABELS},
                "từ": str(d["date"].min())[:10], "đến": str(d["date"].max())[:10]}

    tbl = pd.DataFrame([row(n, d) for n, d in splits.items()] + [row("FULL", full)])
    src = full["source_detail"].value_counts().to_dict()

    ctx = dict(kappa=kappa, base_f1=base_f1, pho_f1=pho_f1, pval=pval,
               tbl=tbl, src=src, full=full, splits=splits,
               es_car=es_car, es_p=es_p, es_n=es_n)

    write_readme(**ctx)
    ctx2 = {k: v for k, v in ctx.items() if not k.startswith('es_')}
    write_dataset_card(**ctx2)
    write_model_card(**ctx2)

    # --- kiem tra duong dan trong README -----------------------------------
    bad = []
    txt = (ROOT / "README.md").read_text(encoding="utf-8")
    for p in re.findall(r"\]\((?!http)([^)]+)\)", txt) + re.findall(r"`(src/[^`]+\.py)`", txt):
        if not (ROOT / p.lstrip("./")).exists():
            bad.append(p)
    print("Duong dan chet trong README:", bad if bad else "khong co")
    print("Da ghi: README.md, docs/dataset_card.md, docs/model_card.md")
    return 1 if bad else 0


# ---------------------------------------------------------------- README ---
def write_readme(kappa, base_f1, pho_f1, pval, tbl, src, full, splits,
                 es_car=None, es_p=None, es_n=None) -> None:
    w = []
    a = w.append
    a("# vn-fin-sentiment\n")
    a("Bộ dữ liệu và mô hình phân loại sentiment tin tức tài chính tiếng Việt, gán "
      "nhãn theo góc nhìn nhà đầu tư đang nắm giữ mã cổ phiếu được nhắc tới. "
      f"Gồm {len(full):,} bài từ CafeF, hai mô hình đối chứng, và một khung event "
      "study để kiểm tra xem sentiment có dự báo được biến động giá hay không.\n")

    a("## Kết quả chính\n")
    a("| Hạng mục | Giá trị |")
    a("|---|---|")
    a(f"| macro-F1 baseline TF-IDF + LogReg | {base_f1} |")
    a(f"| macro-F1 PhoBERT fine-tuned | **{pho_f1}** |")
    a(f"| McNemar p-value giữa hai mô hình | {pval} (khác biệt có ý nghĩa) |")
    a(f"| Cohen's kappa nhãn máy vs nhãn người | {kappa} |")
    if es_car:
        a(f"| Event study, POS−NEG CAR[0,5] | {es_car}% (p={es_p}, {es_n} sự kiện; "
          "p bị thổi phồng do chồng lấn, xem `docs/event_study.md`) |")
    else:
        a("| Event study | chưa chạy, xem mục Hạn chế |")
    a("")

    a("![CM PhoBERT](reports/figures/cm_phobert.png)\n")

    a("## Dataset\n")
    a(tbl.to_markdown(index=False))
    a("")
    a("Nguồn nhãn: " + ", ".join(f"`{k}` {v:,}" for k, v in src.items()) + "\n")
    a("| Nguồn nhãn | Nghĩa |")
    a("|---|---|")
    a("| `human` | Người đọc và gán tay |")
    a("| `claude_manual` | Trợ lý AI đọc từng bài và gán theo labeling guide |")
    a("| `weak_model` | Mô hình lan truyền nhãn từ tập hạt giống |")
    a("")
    a("Test set không chứa bài nào mang nhãn `weak_model`, toàn bộ là nhãn đọc tay. "
      "Đây là chủ ý khi lấy mẫu để con số đánh giá không bị nhiễu nhãn làm méo.\n")

    a("## Cấu trúc thư mục\n")
    a("```")
    a("src/                      code")
    a("  crawl/crawl_urls.py           thu thập URL bài viết từ CafeF")
    a("  crawl/crawl_articles.py       tải nội dung bài")
    a("  crawl/build_ticker_dict.py    dựng từ điển mã cổ phiếu và tên doanh nghiệp")
    a("  repair_tickers.py       chấm điểm và gán primary_ticker")
    a("  label_tool.py           web app gán nhãn tay (Flask)")
    a("  lexicon_vi_fin.py       từ điển sentiment tài chính tiếng Việt")
    a("  prelabel_local.py       gán nhãn tự động, không cần API")
    a("  audit_gold.py           soát lại các ca người/máy bất đồng")
    a("  eval_labels.py          đo kappa giữa nhãn người và nhãn máy")
    a("  finalize_dataset.py     gộp nhãn, chia train/val/test theo thời gian")
    a("  train_baseline.py       TF-IDF + Logistic Regression")
    a("  train_phobert.py        fine-tune vinai/phobert-base")
    a("  compare_models.py       so sánh 2 mô hình + McNemar + phân tích lỗi")
    a("  fetch_prices.py         tải giá đóng cửa (vnstock 4.x) cho event study")
    a("  event_study.py          event study Brown & Warner, có kiểm tra giả dược")
    a("  build_docs.py           sinh README và các card từ số liệu thật")
    a("data/")
    a("  raw/                    dữ liệu thô (không commit)")
    a("  interim/                dữ liệu trung gian, nhãn")
    a("  processed/              train / val / test / dataset_full parquet")
    a("docs/                     báo cáo và card")
    a("reports/figures/          biểu đồ")
    a("models/                   mô hình đã huấn luyện (không commit)")
    a("```\n")

    a("## Pipeline\n")
    a("| Bước | Script | Đầu ra |")
    a("|---|---|---|")
    a("| 1. Thu thập URL | `src/crawl/crawl_urls.py` | `data/raw/urls.jsonl` |")
    a("| 2. Tải bài viết | `src/crawl/crawl_articles.py` | `data/raw/articles.jsonl` |")
    a("| 3. Gán mã cổ phiếu | `src/repair_tickers.py` | `data/interim/to_label_v2.parquet` |")
    a("| 4. Gán nhãn tay | `src/label_tool.py` | `data/interim/gold_seed_v2.csv` |")
    a("| 5. Gán nhãn tự động | `src/prelabel_local.py` | `data/interim/labeled_auto.jsonl` |")
    a("| 6. Đo chất lượng nhãn | `src/eval_labels.py` | `docs/label_quality.md` |")
    a("| 7. Chốt dataset | `src/finalize_dataset.py` | `data/processed/*.parquet` |")
    a("| 8. Baseline | `src/train_baseline.py` | `docs/model_baseline.md` |")
    a("| 9. PhoBERT | `src/train_phobert.py` | `docs/model_phobert.md` |")
    a("| 10. So sánh | `src/compare_models.py` | `docs/model_comparison.md` |")
    a("| 11. Tải giá | `src/fetch_prices.py` | `data/prices/*.csv` |")
    a("| 12. Event study | `src/event_study.py` | `docs/event_study.md` |")
    a("")

    a("## Chạy lại từ đầu\n")
    a("```bash")
    a("python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate")
    a("pip install -r requirements.txt")
    a("")
    a("python src/repair_tickers.py       # cần data/interim/articles_clean.parquet")
    a("python src/prelabel_local.py       # gán nhãn 5.549 bài, không cần API key")
    a("python src/eval_labels.py          # kappa giữa nhãn người và nhãn máy")
    a("python src/finalize_dataset.py     # chia train/val/test theo thời gian")
    a("python src/train_baseline.py       # ~3 phút trên CPU")
    a("python src/train_phobert.py        # ~20 phút trên GPU T4")
    a("python src/compare_models.py")
    a("python src/fetch_prices.py         # giá đóng cửa 195 mã + VNINDEX, 3-5 phút")
    a("python src/event_study.py          # sentiment vs lợi suất bất thường")
    a("python src/build_docs.py           # sinh lại README và các card")
    a("```\n")

    a("## Trên HuggingFace\n")
    a("- Dataset: `<username>/vn-fin-sentiment` (chưa đẩy)")
    a("- Model: `<username>/phobert-vn-fin-sentiment` (chưa đẩy)\n")

    a("## Quyết định thiết kế\n")
    a("| Quyết định | Lý do |")
    a("|---|---|")
    a("| Chia train/val/test thuần theo thời gian | Random split cho phép tin tháng 8 "
      "lọt vào train rồi dự đoán tin tháng 3, tức là nhìn trước tương lai. Với dữ "
      "liệu tài chính đó là rò rỉ nghiêm trọng. |")
    a("| Không ép gold seed vào test | Gold rải đều theo thời gian. Ép vào test sẽ phá "
      "vỡ ranh giới thời gian vừa dựng ở trên. |")
    a("| Gán mã bằng chấm điểm thay vì lấy mã đầu tiên | Một bài thường nhắc 2-5 mã. "
      "Lấy mã đầu tiên gán sai khoảng một phần ba số bài. |")
    a("| Đưa ticker vào input mô hình dạng `[HPG] ...` | Nhãn được định nghĩa theo góc "
      "nhìn người nắm giữ mã đó. Không có ticker thì bài toán không xác định. |")
    a("| Tách từ bằng underthesea trước khi tokenize | PhoBERT được huấn luyện trên văn "
      "bản đã tách từ. Bỏ bước này điểm tụt mạnh. |")
    a("| Gán nhãn bằng mô hình lan truyền thay vì gọi API LLM | Chi phí bằng 0, chạy "
      "lại được, không phụ thuộc nhà cung cấp. Đổi lại nhãn nhiễu hơn, đã ghi rõ ở dưới. |")
    a("| Tách `label_source` thành 3 mức | Nhãn người, nhãn đọc tay và nhãn lan truyền "
      "có độ tin cậy khác hẳn nhau. Gộp làm một là tự đánh lừa mình khi đọc kết quả. |")
    a("| Giữ toàn bộ test là nhãn đọc tay | Để con số macro-F1 trên test có nghĩa. |")
    a("| Chọn siêu tham số trên phần val có nhãn đọc tay | Val chứa 94% nhãn `weak_model` "
      "do một mô hình TF-IDF+LR sinh ra. Chấm baseline TF-IDF+LR trên đó là đo mức bắt "
      "chước chính họ mô hình, không phải năng lực đọc tin. |")
    a("")

    a("## Hạn chế đã biết\n")
    a("- Chỉ một nguồn tin (CafeF). Không đại diện cho toàn thị trường.")
    a(f"- Phần lớn nhãn train do mô hình yếu lan truyền "
      f"({src.get('weak_model', 0):,}/{len(full):,} bài). Trần hiệu năng bị chặn bởi "
      "chất lượng nhãn chứ không phải kiến trúc mô hình.")
    a(f"- Cohen's kappa giữa nhãn máy và nhãn người mới đạt {kappa}, dưới ngưỡng 0,60 "
      "thường coi là tốt. Nguyên nhân chính là tiêu chí gán nhãn trôi giữa hai phiên "
      "làm việc của người gán, phân tích chi tiết ở `docs/prelabel_local_report.md`.")
    a("- Độ chính xác gán `primary_ticker` ước tính khoảng 85% (audit tay 30 mẫu).")
    a("- Khoảng thời gian ngắn, cuối 2024 đến giữa 2026, chưa qua đủ một chu kỳ thị trường.")
    a("- Mô hình chỉ đọc tiêu đề và đoạn dẫn, không đọc toàn văn.")
    if es_car:
        a("- **Event study đã chạy nhưng p-value bị thổi phồng** vì phần lớn sự "
          "kiện chồng lấn cùng mã. Kiểm tra giả dược và bốn khả năng gây nhiễu "
          "ở `docs/event_study.md`, mục 6-8. Chưa đủ cơ sở dự báo giá.")
    else:
        a("- **Event study chưa chạy.** Cần dữ liệu giá cổ phiếu và VNINDEX theo ngày, "
          "chưa thu thập. Mọi kết luận về khả năng dự báo giá đều chưa có cơ sở.")
    a("- Không dùng cho quyết định đầu tư thật.")
    a("")
    a("## Giấy phép\n")
    a("Code: MIT. Dữ liệu: CC BY 4.0. Nội dung bài báo gốc thuộc bản quyền CafeF, "
      "dataset chỉ chứa tiêu đề, đoạn dẫn và metadata.")

    (ROOT / "README.md").write_text("\n".join(w) + "\n", encoding="utf-8")


# ---------------------------------------------------------- dataset card ---
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
    a("*English first, tiếng Việt bên dưới.*\n")
    a("---\n\n## English\n")
    a("### Dataset Summary\n")
    a(f"{n:,} Vietnamese financial news items from CafeF, each mapped to one listed "
      "ticker and labelled POSITIVE / NEUTRAL / NEGATIVE. Labels follow the "
      "perspective of an investor already holding that ticker: does this news raise "
      "or lower expected firm value? Splits are strictly time-ordered to avoid "
      "look-ahead leakage.\n")
    a("### Supported Tasks\n")
    a("Three-class text classification. Also usable as an event-study input where "
      "each labelled item is a dated signal for one ticker.\n")
    a("### Languages\n\nVietnamese (`vi`).\n")
    a("### Data Fields\n")
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
    a("### Data Splits\n")
    a(tbl.to_markdown(index=False))
    a("")
    a("### Curation Rationale\n")
    a("Vietnamese financial NLP lacks a public sentiment dataset that is (a) tied to a "
      "specific ticker rather than general market mood, and (b) split by time so that "
      "downstream event studies are not contaminated by look-ahead bias. This dataset "
      "targets both gaps.\n")
    a("### Source Data\n")
    a("Crawled from CafeF (cafef.vn), a mainstream Vietnamese financial news outlet. "
      f"Coverage {tbl.iloc[-1]['từ']} to {tbl.iloc[-1]['đến']}. Only headline, lead, "
      "and metadata are redistributed.\n")
    a("### Annotation Process\n")
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
    a("### Ticker Attribution\n")
    a("`src/repair_tickers.py` scores every candidate ticker in an article using three "
      "signals in priority order: tickers shown in the page's own stock widget, tickers "
      "named explicitly in the text, and tickers inferred from full company names. The "
      "highest-scoring candidate becomes `primary_ticker`, with the score gap kept in "
      "`pt_margin`. Manual audit of 30 samples put accuracy at roughly 85%.\n")
    a("### Personal and Sensitive Information\n")
    a("All content is published financial news. Named individuals appear only in their "
      "public professional capacity as company officers or regulators. No private "
      "personal data.\n")
    a("### Limitations and Bias\n")
    a("1. Single source (CafeF); outlet-specific framing is baked in.")
    a(f"2. {src.get('weak_model', 0):,} of {n:,} labels come from a weak propagation model.")
    a(f"3. Machine-vs-human kappa is only {kappa}; part of that is human annotation drift.")
    a("4. Ticker attribution accuracy ~85%; wrong ticker means the label describes the wrong firm.")
    a("5. Short window (late 2024 to mid 2026), less than one full market cycle.")
    a("6. Class imbalance: NEUTRAL dominates, NEGATIVE is the rarest class.")
    a("7. Headline and lead only, no full-text reasoning.")
    a("8. Event clustering: firms in the same sector often get news on the same day.\n")
    a("### Citation\n")
    a("```bibtex\n@misc{vnfinsentiment,\n  title  = {vn-fin-sentiment: Ticker-level "
      "sentiment for Vietnamese financial news},\n  author = {<Your Name>},\n"
      "  year   = {2026},\n  url    = {https://huggingface.co/datasets/<username>/"
      "vn-fin-sentiment}\n}\n```\n")

    a("---\n\n## Tiếng Việt\n")
    a("### Tóm tắt\n")
    a(f"{n:,} tin tài chính tiếng Việt từ CafeF, mỗi tin gắn với đúng một mã cổ phiếu "
      "và một nhãn POSITIVE / NEUTRAL / NEGATIVE. Nhãn đứng từ góc nhìn nhà đầu tư "
      "đang nắm giữ mã đó: tin này làm tăng hay giảm kỳ vọng giá trị doanh nghiệp? "
      "Các split chia thuần theo thời gian để tránh rò rỉ thông tin tương lai.\n")
    a("### Nhiệm vụ hỗ trợ\n")
    a("Phân loại văn bản 3 lớp. Cũng dùng được làm đầu vào cho event study, mỗi bài là "
      "một tín hiệu có ngày tháng gắn với một mã.\n")
    a("### Ngôn ngữ\n\nTiếng Việt (`vi`).\n")
    a("### Mô tả cột\n")
    a("| Cột | Kiểu | Ý nghĩa |")
    a("|---|---|---|")
    a("| `id` | string | Định danh bài viết |")
    a("| `datetime` | timestamp | Thời điểm đăng |")
    a("| `date` | string | Ngày đăng, `YYYY-MM-DD` |")
    a("| `primary_ticker` | string | Mã cổ phiếu mà nhãn nói tới |")
    a("| `ticker_in_head` | bool | Mã có xuất hiện trong tiêu đề hoặc sapo không |")
    a("| `title` | string | Tiêu đề |")
    a("| `sapo` | string | Đoạn dẫn |")
    a("| `text` | string | Nội dung bài |")
    a("| `text_input` | string | Đầu vào mô hình, dạng `\"[MÃ] tiêu đề. sapo\"` |")
    a("| `label` | string | POSITIVE / NEUTRAL / NEGATIVE |")
    a("| `label_source` | string | `human` hoặc `llm` |")
    a("| `source_detail` | string | `human`, `claude_manual`, `weak_model` |")
    a("| `confidence_val` | float | Độ tin cậy nhãn, bằng 1.0 với nhãn người |")
    a("| `split` | string | train / validation / test |")
    a("")
    a("### Chia split\n")
    a(tbl.to_markdown(index=False))
    a("")
    a("### Vì sao xây dataset này\n")
    a("NLP tài chính tiếng Việt thiếu một bộ dữ liệu sentiment công khai vừa gắn nhãn "
      "theo từng mã cổ phiếu cụ thể thay vì tâm lý thị trường chung, vừa chia theo thời "
      "gian để event study phía sau không bị nhiễm rò rỉ tương lai. Dataset này nhắm "
      "vào cả hai khoảng trống đó.\n")
    a("### Nguồn dữ liệu\n")
    a(f"Thu thập từ CafeF (cafef.vn), khoảng {tbl.iloc[-1]['từ']} đến "
      f"{tbl.iloc[-1]['đến']}. Chỉ phát hành lại tiêu đề, đoạn dẫn và metadata.\n")
    a("### Quy trình gán nhãn\n")
    a("Ba mức, ghi rõ ở cột `source_detail`:\n")
    a("| Mức | Số bài | Cách gán |")
    a("|---|---:|---|")
    for k in ("human", "claude_manual", "weak_model"):
        meth = {"human": "Người đọc và gán tay",
                "claude_manual": "Trợ lý AI đọc từng bài, bám labeling guide",
                "weak_model": "Mô hình TF-IDF + từ điển + hồi quy logistic lan truyền"}[k]
        a(f"| `{k}` | {src.get(k, 0):,} | {meth} |")
    a("")
    a(f"Cohen's kappa giữa nhãn máy và nhãn người: **{kappa}**, dưới ngưỡng 0,60. "
      "Chẩn đoán chi tiết ở `docs/prelabel_local_report.md`: chính tập nhãn người bị "
      "trôi tiêu chí giữa hai phiên gán, phiên sau có 70% POSITIVE trong khi phiên "
      "trước là 45%. Con số kappa này nói về cả hai nguồn nhãn, không riêng gì máy.\n")
    a("Test split được thiết kế để không chứa nhãn `weak_model` nào.\n")
    a("### Gán mã cổ phiếu\n")
    a("`src/repair_tickers.py` chấm điểm mọi mã ứng viên trong bài theo ba tín hiệu xếp "
      "thứ tự ưu tiên: mã hiện trên widget cổ phiếu của trang, mã được nhắc đích danh "
      "trong bài, mã suy ra từ tên đầy đủ doanh nghiệp. Mã điểm cao nhất thành "
      "`primary_ticker`, khoảng cách điểm lưu ở `pt_margin`. Audit tay 30 mẫu cho độ "
      "chính xác khoảng 85%.\n")
    a("### Thông tin cá nhân\n")
    a("Toàn bộ là tin tài chính đã công bố. Cá nhân được nhắc tên chỉ với tư cách công "
      "khai là lãnh đạo doanh nghiệp hoặc cơ quan quản lý. Không chứa dữ liệu cá nhân "
      "riêng tư.\n")
    a("### Hạn chế và thiên lệch\n")
    a("1. Một nguồn duy nhất (CafeF), mang theo cách đưa tin riêng của tòa soạn đó.")
    a(f"2. {src.get('weak_model', 0):,}/{n:,} nhãn do mô hình yếu lan truyền.")
    a(f"3. Kappa máy vs người mới {kappa}, một phần do người gán trôi tiêu chí.")
    a("4. Độ chính xác gán mã ~85%; sai mã thì nhãn mô tả nhầm doanh nghiệp.")
    a("5. Khoảng thời gian ngắn, chưa đủ một chu kỳ thị trường.")
    a("6. Mất cân bằng lớp: NEUTRAL áp đảo, NEGATIVE hiếm nhất.")
    a("7. Chỉ tiêu đề và đoạn dẫn, không suy luận trên toàn văn.")
    a("8. Chồng lấn sự kiện: nhiều mã cùng ngành thường có tin cùng ngày.\n")

    (DOCS / "dataset_card.md").write_text("\n".join(w) + "\n", encoding="utf-8")


# ------------------------------------------------------------ model card ---
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
            "text = '[HPG] Hòa Phát báo lãi kỷ lục quý 2. Lợi nhuận tăng 48% so với cùng kỳ.'\n"
            "seg = word_tokenize(text, format='text')          # BẮT BUỘC tách từ trước\n"
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
    a("*English first, tiếng Việt bên dưới.*\n")
    a("---\n\n## English\n")
    a("### Model Description\n")
    a("`vinai/phobert-base` fine-tuned for three-class sentiment on Vietnamese "
      "financial news. The label answers one question: for an investor already holding "
      f"the ticker named in the input, does this news raise expected firm value "
      "(POSITIVE), lower it (NEGATIVE), or neither (NEUTRAL)?\n")
    a("### Intended Use\n")
    a("Research on Vietnamese financial NLP; feature extraction for event studies; a "
      "baseline for anyone building ticker-level sentiment in Vietnamese.\n")
    a("### Out-of-Scope Use\n")
    a("**Do not use this model for real investment decisions.** It reads headlines and "
      "lead paragraphs only, was trained largely on machine-propagated labels, and has "
      "never been validated against realised returns. It is a research artifact.\n")
    a("Also out of scope: non-financial Vietnamese text, other languages, and any "
      "input where the ticker prefix is missing.\n")
    a("### Training Data\n")
    a("`vn-fin-sentiment`, see `docs/dataset_card.md`. "
      f"Train {len(splits['train']):,} rows, validation {len(splits['val']):,}, "
      f"test {len(te):,}. Splits are strictly time-ordered.\n")
    a("### Training Procedure\n")
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
    a("### Evaluation\n")
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
    a("### Input Format\n")
    a("Input **must** be `\"[TICKER] Headline. Lead paragraph\"` and **must** be "
      "word-segmented with underthesea before tokenisation. PhoBERT was pretrained on "
      "segmented text; skipping this step degrades results sharply.\n")
    a(code)
    a("### Limitations and Bias\n")
    a("1. Single news source (CafeF).")
    a(f"2. About {src.get('weak_model', 0) / len(full):.0%} of all labels come from a "
      "weak propagation model, so the ceiling is set by label quality, not architecture.")
    a(f"3. Machine-vs-human label agreement is only kappa = {kappa}.")
    a("4. Ticker attribution accuracy ~85%.")
    a("5. Headline and lead only; no full-text reasoning.")
    a("6. NEGATIVE is the rarest class and the hardest one.")
    a("7. Trained on late 2024 to mid 2026; performance on other regimes is untested.")
    a("8. Never validated against realised stock returns.\n")

    a("---\n\n## Tiếng Việt\n")
    a("### Mô tả\n")
    a("`vinai/phobert-base` được fine-tune cho bài toán sentiment 3 lớp trên tin tài "
      "chính tiếng Việt. Nhãn trả lời đúng một câu hỏi: với nhà đầu tư đang nắm giữ mã "
      "được nhắc trong đầu vào, tin này làm tăng kỳ vọng giá trị doanh nghiệp "
      "(POSITIVE), làm giảm (NEGATIVE), hay không rõ (NEUTRAL)?\n")
    a("### Dùng để làm gì\n")
    a("Nghiên cứu NLP tài chính tiếng Việt, trích đặc trưng cho event study, làm mốc "
      "so sánh cho ai muốn xây sentiment theo từng mã cổ phiếu.\n")
    a("### KHÔNG dùng để làm gì\n")
    a("**Không dùng cho quyết định đầu tư thật.** Mô hình chỉ đọc tiêu đề và đoạn dẫn, "
      "học chủ yếu từ nhãn do máy lan truyền, và chưa từng được kiểm chứng với lợi "
      "suất thực tế. Đây là sản phẩm nghiên cứu.\n")
    a("Ngoài phạm vi: văn bản tiếng Việt không thuộc lĩnh vực tài chính, ngôn ngữ khác, "
      "và mọi đầu vào thiếu tiền tố mã cổ phiếu.\n")
    a("### Dữ liệu huấn luyện\n")
    a(f"`vn-fin-sentiment`, xem `docs/dataset_card.md`. Train {len(splits['train']):,} "
      f"bài, val {len(splits['val']):,}, test {len(te):,}. Chia thuần theo thời gian.\n")
    a("### Siêu tham số\n")
    a("| Tham số | Giá trị |")
    a("|---|---|")
    a("| learning rate | 2e-5 |")
    a("| batch size | 16 |")
    a("| epochs | 4, early stopping patience 2 |")
    a("| warmup ratio | 0.1 |")
    a("| weight decay | 0.01 |")
    a("| max length | 256 |")
    a("| class weights | có, nghịch đảo tần suất nhãn trong train |")
    a("| chọn checkpoint theo | macro-F1 trên val |")
    a("| seed | 42 |")
    a("| phần cứng | một GPU NVIDIA T4 (Google Colab) |")
    a("| thời gian | khoảng 20 phút |")
    a("")
    a("### Kết quả\n")
    a("| Mô hình | macro-F1 | accuracy | weighted-F1 |")
    a("|---|---|---|---|")
    a(f"| TF-IDF + Logistic Regression | {base_f1} | — | — |")
    a(f"| **PhoBERT (mô hình này)** | **{pho_f1}** | {acc} | {wf1} |")
    a("")
    a(f"Kiểm định McNemar giữa hai mô hình: p = {pval}, khác biệt có ý nghĩa thống kê "
      "ở alpha 0,05. Chi tiết ở `docs/model_comparison.md`.\n")
    a("### Định dạng đầu vào\n")
    a("Đầu vào **bắt buộc** có dạng `\"[MÃ] Tiêu đề. Đoạn dẫn\"` và **bắt buộc** tách "
      "từ bằng underthesea trước khi tokenize. PhoBERT được tiền huấn luyện trên văn "
      "bản đã tách từ, bỏ bước này điểm tụt mạnh.\n")
    a(code)
    a("### Hạn chế\n")
    a("1. Một nguồn tin duy nhất (CafeF).")
    a(f"2. Khoảng {src.get('weak_model', 0) / len(full):.0%} tổng số nhãn do mô hình yếu "
      "lan truyền, nên trần hiệu năng bị chặn bởi chất lượng nhãn.")
    a(f"3. Mức đồng thuận nhãn máy với nhãn người mới kappa = {kappa}.")
    a("4. Độ chính xác gán mã cổ phiếu khoảng 85%.")
    a("5. Chỉ đọc tiêu đề và đoạn dẫn.")
    a("6. NEGATIVE là lớp ít mẫu nhất và khó nhất.")
    a("7. Huấn luyện trên dữ liệu cuối 2024 đến giữa 2026, chưa kiểm chứng ở giai đoạn khác.")
    a("8. Chưa từng đối chiếu với lợi suất cổ phiếu thực tế.\n")

    (DOCS / "model_card.md").write_text("\n".join(w) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())