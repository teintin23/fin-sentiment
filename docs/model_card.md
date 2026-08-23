---
language:
- vi
license: mit
base_model: vinai/phobert-base
pipeline_tag: text-classification
tags:
- finance
- vietnamese
- sentiment
- phobert
---

# phobert-vn-fin-sentiment

*English first, tiếng Việt bên dưới.*

---

## English

### Model Description

`vinai/phobert-base` fine-tuned for three-class sentiment on Vietnamese financial news. The label answers one question: for an investor already holding the ticker named in the input, does this news raise expected firm value (POSITIVE), lower it (NEGATIVE), or neither (NEUTRAL)?

### Intended Use

Research on Vietnamese financial NLP; feature extraction for event studies; a baseline for anyone building ticker-level sentiment in Vietnamese.

### Out-of-Scope Use

**Do not use this model for real investment decisions.** It reads headlines and lead paragraphs only, was trained largely on machine-propagated labels, and has never been validated against realised returns. It is a research artifact.

Also out of scope: non-financial Vietnamese text, other languages, and any input where the ticker prefix is missing.

### Training Data

`vn-fin-sentiment`, see `docs/dataset_card.md`. Train 3,777 rows, validation 810, test 810. Splits are strictly time-ordered.

### Training Procedure

| Hyperparameter | Value |
|---|---|
| learning rate | 2e-5 |
| batch size | 16 |
| epochs | 4, early stopping patience 2 |
| warmup ratio | 0.1 |
| weight decay | 0.01 |
| max length | 256 |
| class weights | yes, inverse label frequency in train |
| best-model metric | macro-F1 on validation |
| seed | 42 |
| hardware | single NVIDIA T4 (Google Colab) |
| wall time | roughly 20 minutes |

### Evaluation

| Model | macro-F1 | accuracy | weighted-F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.7483 | — | — |
| **PhoBERT (this model)** | **0.7948** | 0.8000 | 0.8002 |

McNemar test between the two: p = 1.331e-02, so the gap is statistically significant at alpha 0.05. Full breakdown in `docs/model_comparison.md`.

Confusion matrix, rows are true labels:

| | NEGATIVE | NEUTRAL | POSITIVE |
|---|---|---|---|
| **NEGATIVE** | 108 | 26 | 6 |
| **NEUTRAL** | 19 | 330 | 60 |
| **POSITIVE** | 8 | 43 | 210 |

![confusion matrix](../reports/figures/cm_phobert.png)

### Input Format

Input **must** be `"[TICKER] Headline. Lead paragraph"` and **must** be word-segmented with underthesea before tokenisation. PhoBERT was pretrained on segmented text; skipping this step degrades results sharply.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
import torch

REPO = 'models/phobert-vnfin'   # or '<username>/phobert-vn-fin-sentiment'
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

text = '[HPG] Hòa Phát báo lãi kỷ lục quý 2. Lợi nhuận tăng 48% so với cùng kỳ.'
seg = word_tokenize(text, format='text')          # BẮT BUỘC tách từ trước
x = tok(seg, return_tensors='pt', truncation=True, max_length=256)
with torch.no_grad():
    print(model.config.id2label[model(**x).logits.argmax(-1).item()])
```

### Limitations and Bias

1. Single news source (CafeF).
2. About 76% of all labels come from a weak propagation model, so the ceiling is set by label quality, not architecture.
3. Machine-vs-human label agreement is only kappa = 0.5505.
4. Ticker attribution accuracy ~85%.
5. Headline and lead only; no full-text reasoning.
6. NEGATIVE is the rarest class and the hardest one.
7. Trained on late 2024 to mid 2026; performance on other regimes is untested.
8. Never validated against realised stock returns.

---

## Tiếng Việt

### Mô tả

`vinai/phobert-base` được fine-tune cho bài toán sentiment 3 lớp trên tin tài chính tiếng Việt. Nhãn trả lời đúng một câu hỏi: với nhà đầu tư đang nắm giữ mã được nhắc trong đầu vào, tin này làm tăng kỳ vọng giá trị doanh nghiệp (POSITIVE), làm giảm (NEGATIVE), hay không rõ (NEUTRAL)?

### Dùng để làm gì

Nghiên cứu NLP tài chính tiếng Việt, trích đặc trưng cho event study, làm mốc so sánh cho ai muốn xây sentiment theo từng mã cổ phiếu.

### KHÔNG dùng để làm gì

**Không dùng cho quyết định đầu tư thật.** Mô hình chỉ đọc tiêu đề và đoạn dẫn, học chủ yếu từ nhãn do máy lan truyền, và chưa từng được kiểm chứng với lợi suất thực tế. Đây là sản phẩm nghiên cứu.

Ngoài phạm vi: văn bản tiếng Việt không thuộc lĩnh vực tài chính, ngôn ngữ khác, và mọi đầu vào thiếu tiền tố mã cổ phiếu.

### Dữ liệu huấn luyện

`vn-fin-sentiment`, xem `docs/dataset_card.md`. Train 3,777 bài, val 810, test 810. Chia thuần theo thời gian.

### Siêu tham số

| Tham số | Giá trị |
|---|---|
| learning rate | 2e-5 |
| batch size | 16 |
| epochs | 4, early stopping patience 2 |
| warmup ratio | 0.1 |
| weight decay | 0.01 |
| max length | 256 |
| class weights | có, nghịch đảo tần suất nhãn trong train |
| chọn checkpoint theo | macro-F1 trên val |
| seed | 42 |
| phần cứng | một GPU NVIDIA T4 (Google Colab) |
| thời gian | khoảng 20 phút |

### Kết quả

| Mô hình | macro-F1 | accuracy | weighted-F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.7483 | — | — |
| **PhoBERT (mô hình này)** | **0.7948** | 0.8000 | 0.8002 |

Kiểm định McNemar giữa hai mô hình: p = 1.331e-02, khác biệt có ý nghĩa thống kê ở alpha 0,05. Chi tiết ở `docs/model_comparison.md`.

### Định dạng đầu vào

Đầu vào **bắt buộc** có dạng `"[MÃ] Tiêu đề. Đoạn dẫn"` và **bắt buộc** tách từ bằng underthesea trước khi tokenize. PhoBERT được tiền huấn luyện trên văn bản đã tách từ, bỏ bước này điểm tụt mạnh.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
import torch

REPO = 'models/phobert-vnfin'   # or '<username>/phobert-vn-fin-sentiment'
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

text = '[HPG] Hòa Phát báo lãi kỷ lục quý 2. Lợi nhuận tăng 48% so với cùng kỳ.'
seg = word_tokenize(text, format='text')          # BẮT BUỘC tách từ trước
x = tok(seg, return_tensors='pt', truncation=True, max_length=256)
with torch.no_grad():
    print(model.config.id2label[model(**x).logits.argmax(-1).item()])
```

### Hạn chế

1. Một nguồn tin duy nhất (CafeF).
2. Khoảng 76% tổng số nhãn do mô hình yếu lan truyền, nên trần hiệu năng bị chặn bởi chất lượng nhãn.
3. Mức đồng thuận nhãn máy với nhãn người mới kappa = 0.5505.
4. Độ chính xác gán mã cổ phiếu khoảng 85%.
5. Chỉ đọc tiêu đề và đoạn dẫn.
6. NEGATIVE là lớp ít mẫu nhất và khó nhất.
7. Huấn luyện trên dữ liệu cuối 2024 đến giữa 2026, chưa kiểm chứng ở giai đoạn khác.
8. Chưa từng đối chiếu với lợi suất cổ phiếu thực tế.

