# vn-fin-sentiment

Bộ dữ liệu và mô hình phân loại sentiment tin tức tài chính tiếng Việt, gán nhãn theo góc nhìn nhà đầu tư đang nắm giữ mã cổ phiếu được nhắc tới. Gồm 5,397 bài từ CafeF, hai mô hình đối chứng, và một khung event study để kiểm tra xem sentiment có dự báo được biến động giá hay không.

## Kết quả chính

| Hạng mục | Giá trị |
|---|---|
| macro-F1 baseline TF-IDF + LogReg | 0.7483 |
| macro-F1 PhoBERT fine-tuned | **0.7948** |
| McNemar p-value giữa hai mô hình | 1.331e-02 (khác biệt có ý nghĩa) |
| Cohen's kappa nhãn máy vs nhãn người | 0.5505 |
| Event study | chưa chạy, xem mục Hạn chế |

![CM PhoBERT](reports/figures/cm_phobert.png)

## Dataset

| split   |    n |   NEGATIVE |   NEUTRAL |   POSITIVE | từ         | đến        |
|:--------|-----:|-----------:|----------:|-----------:|:-----------|:-----------|
| train   | 3777 |        389 |      2337 |       1051 | 2024-12-16 | 2026-04-21 |
| val     |  810 |         92 |       529 |        189 | 2026-04-21 | 2026-06-19 |
| test    |  810 |        140 |       409 |        261 | 2026-06-19 | 2026-08-18 |
| FULL    | 5397 |        621 |      3275 |       1501 | 2024-12-16 | 2026-08-18 |

Nguồn nhãn: `weak_model` 4,119, `claude_manual` 1,135, `human` 143

| Nguồn nhãn | Nghĩa |
|---|---|
| `human` | Người đọc và gán tay |
| `claude_manual` | Trợ lý AI đọc từng bài và gán theo labeling guide |
| `weak_model` | Mô hình lan truyền nhãn từ tập hạt giống |

Test set không chứa bài nào mang nhãn `weak_model`, toàn bộ là nhãn đọc tay. Đây là chủ ý khi lấy mẫu để con số đánh giá không bị nhiễu nhãn làm méo.

## Cấu trúc thư mục

```
src/                      code
  crawl/crawl_urls.py           thu thập URL bài viết từ CafeF
  crawl/crawl_articles.py       tải nội dung bài
  crawl/build_ticker_dict.py    dựng từ điển mã cổ phiếu và tên doanh nghiệp
  repair_tickers.py       chấm điểm và gán primary_ticker
  label_tool.py           web app gán nhãn tay (Flask)
  lexicon_vi_fin.py       từ điển sentiment tài chính tiếng Việt
  prelabel_local.py       gán nhãn tự động, không cần API
  audit_gold.py           soát lại các ca người/máy bất đồng
  eval_labels.py          đo kappa giữa nhãn người và nhãn máy
  finalize_dataset.py     gộp nhãn, chia train/val/test theo thời gian
  train_baseline.py       TF-IDF + Logistic Regression
  train_phobert.py        fine-tune vinai/phobert-base
  compare_models.py       so sánh 2 mô hình + McNemar + phân tích lỗi
  build_docs.py           sinh README và các card từ số liệu thật
data/
  raw/                    dữ liệu thô (không commit)
  interim/                dữ liệu trung gian, nhãn
  processed/              train / val / test / dataset_full parquet
docs/                     báo cáo và card
reports/figures/          biểu đồ
models/                   mô hình đã huấn luyện (không commit)
```

## Pipeline

| Bước | Script | Đầu ra |
|---|---|---|
| 1. Thu thập URL | `src/crawl/crawl_urls.py` | `data/raw/urls.jsonl` |
| 2. Tải bài viết | `src/crawl/crawl_articles.py` | `data/raw/articles.jsonl` |
| 3. Gán mã cổ phiếu | `src/repair_tickers.py` | `data/interim/to_label_v2.parquet` |
| 4. Gán nhãn tay | `src/label_tool.py` | `data/interim/gold_seed_v2.csv` |
| 5. Gán nhãn tự động | `src/prelabel_local.py` | `data/interim/labeled_auto.jsonl` |
| 6. Đo chất lượng nhãn | `src/eval_labels.py` | `docs/label_quality.md` |
| 7. Chốt dataset | `src/finalize_dataset.py` | `data/processed/*.parquet` |
| 8. Baseline | `src/train_baseline.py` | `docs/model_baseline.md` |
| 9. PhoBERT | `src/train_phobert.py` | `docs/model_phobert.md` |
| 10. So sánh | `src/compare_models.py` | `docs/model_comparison.md` |

## Chạy lại từ đầu

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/repair_tickers.py       # cần data/interim/articles_clean.parquet
python src/prelabel_local.py       # gán nhãn 5.549 bài, không cần API key
python src/eval_labels.py          # kappa giữa nhãn người và nhãn máy
python src/finalize_dataset.py     # chia train/val/test theo thời gian
python src/train_baseline.py       # ~3 phút trên CPU
python src/train_phobert.py        # ~20 phút trên GPU T4
python src/compare_models.py
python src/build_docs.py           # sinh lại README và các card
```

## Trên HuggingFace

- Dataset: `<username>/vn-fin-sentiment` (chưa đẩy)
- Model: `<username>/phobert-vn-fin-sentiment` (chưa đẩy)

## Quyết định thiết kế

| Quyết định | Lý do |
|---|---|
| Chia train/val/test thuần theo thời gian | Random split cho phép tin tháng 8 lọt vào train rồi dự đoán tin tháng 3, tức là nhìn trước tương lai. Với dữ liệu tài chính đó là rò rỉ nghiêm trọng. |
| Không ép gold seed vào test | Gold rải đều theo thời gian. Ép vào test sẽ phá vỡ ranh giới thời gian vừa dựng ở trên. |
| Gán mã bằng chấm điểm thay vì lấy mã đầu tiên | Một bài thường nhắc 2-5 mã. Lấy mã đầu tiên gán sai khoảng một phần ba số bài. |
| Đưa ticker vào input mô hình dạng `[HPG] ...` | Nhãn được định nghĩa theo góc nhìn người nắm giữ mã đó. Không có ticker thì bài toán không xác định. |
| Tách từ bằng underthesea trước khi tokenize | PhoBERT được huấn luyện trên văn bản đã tách từ. Bỏ bước này điểm tụt mạnh. |
| Gán nhãn bằng mô hình lan truyền thay vì gọi API LLM | Chi phí bằng 0, chạy lại được, không phụ thuộc nhà cung cấp. Đổi lại nhãn nhiễu hơn, đã ghi rõ ở dưới. |
| Tách `label_source` thành 3 mức | Nhãn người, nhãn đọc tay và nhãn lan truyền có độ tin cậy khác hẳn nhau. Gộp làm một là tự đánh lừa mình khi đọc kết quả. |
| Giữ toàn bộ test là nhãn đọc tay | Để con số macro-F1 trên test có nghĩa. |
| Chọn siêu tham số trên phần val có nhãn đọc tay | Val chứa 94% nhãn `weak_model` do một mô hình TF-IDF+LR sinh ra. Chấm baseline TF-IDF+LR trên đó là đo mức bắt chước chính họ mô hình, không phải năng lực đọc tin. |

## Hạn chế đã biết

- Chỉ một nguồn tin (CafeF). Không đại diện cho toàn thị trường.
- Phần lớn nhãn train do mô hình yếu lan truyền (4,119/5,397 bài). Trần hiệu năng bị chặn bởi chất lượng nhãn chứ không phải kiến trúc mô hình.
- Cohen's kappa giữa nhãn máy và nhãn người mới đạt 0.5505, dưới ngưỡng 0,60 thường coi là tốt. Nguyên nhân chính là tiêu chí gán nhãn trôi giữa hai phiên làm việc của người gán, phân tích chi tiết ở `docs/prelabel_local_report.md`.
- Độ chính xác gán `primary_ticker` ước tính khoảng 85% (audit tay 30 mẫu).
- Khoảng thời gian ngắn, cuối 2024 đến giữa 2026, chưa qua đủ một chu kỳ thị trường.
- Mô hình chỉ đọc tiêu đề và đoạn dẫn, không đọc toàn văn.
- **Event study chưa chạy.** Cần dữ liệu giá cổ phiếu và VNINDEX theo ngày, chưa thu thập. Mọi kết luận về khả năng dự báo giá đều chưa có cơ sở.
- Không dùng cho quyết định đầu tư thật.

## Giấy phép

Code: MIT. Dữ liệu: CC BY 4.0. Nội dung bài báo gốc thuộc bản quyền CafeF, dataset chỉ chứa tiêu đề, đoạn dẫn và metadata.
