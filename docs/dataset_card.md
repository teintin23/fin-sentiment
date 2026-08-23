---
language:
- vi
license: cc-by-4.0
task_categories:
- text-classification
task_ids:
- sentiment-classification
tags:
- finance
- vietnamese
- sentiment
- stock-market
size_categories:
- 1K<n<10K
---

# vn-fin-sentiment

*English first, tiếng Việt bên dưới.*

---

## English

### Dataset Summary

5,397 Vietnamese financial news items from CafeF, each mapped to one listed ticker and labelled POSITIVE / NEUTRAL / NEGATIVE. Labels follow the perspective of an investor already holding that ticker: does this news raise or lower expected firm value? Splits are strictly time-ordered to avoid look-ahead leakage.

### Supported Tasks

Three-class text classification. Also usable as an event-study input where each labelled item is a dated signal for one ticker.

### Languages

Vietnamese (`vi`).

### Data Fields

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable article identifier |
| `datetime` | timestamp | Publication timestamp |
| `date` | string | Publication date, `YYYY-MM-DD` |
| `primary_ticker` | string | Ticker the label refers to |
| `ticker_in_head` | bool | Whether the ticker appears in title or lead |
| `title` | string | Article headline |
| `sapo` | string | Lead paragraph |
| `text` | string | Body text |
| `text_input` | string | Model input, format `"[TICKER] title. sapo"` |
| `label` | string | POSITIVE / NEUTRAL / NEGATIVE |
| `label_source` | string | `human` or `llm` |
| `source_detail` | string | `human`, `claude_manual`, or `weak_model` |
| `confidence_val` | float | Label confidence, 1.0 for human labels |
| `split` | string | train / validation / test |

### Data Splits

| split   |    n |   NEGATIVE |   NEUTRAL |   POSITIVE | từ         | đến        |
|:--------|-----:|-----------:|----------:|-----------:|:-----------|:-----------|
| train   | 3777 |        389 |      2337 |       1051 | 2024-12-16 | 2026-04-21 |
| val     |  810 |         92 |       529 |        189 | 2026-04-21 | 2026-06-19 |
| test    |  810 |        140 |       409 |        261 | 2026-06-19 | 2026-08-18 |
| FULL    | 5397 |        621 |      3275 |       1501 | 2024-12-16 | 2026-08-18 |

### Curation Rationale

Vietnamese financial NLP lacks a public sentiment dataset that is (a) tied to a specific ticker rather than general market mood, and (b) split by time so that downstream event studies are not contaminated by look-ahead bias. This dataset targets both gaps.

### Source Data

Crawled from CafeF (cafef.vn), a mainstream Vietnamese financial news outlet. Coverage 2024-12-16 to 2026-08-18. Only headline, lead, and metadata are redistributed.

### Annotation Process

Three tiers, recorded per row in `source_detail`:

| Tier | Count | Method |
|---|---:|---|
| `human` | 143 | Read and labelled by a human annotator |
| `claude_manual` | 1,135 | Read individually by an AI assistant following the labelling guide |
| `weak_model` | 4,119 | Propagated by TF-IDF + lexicon + calibrated logistic regression |

Agreement between machine labels and the human gold set: Cohen's kappa = **0.5505** on the overlapping items. This is below the 0.60 threshold usually called substantial. Diagnosis in `docs/prelabel_local_report.md`: the human gold set itself drifted between two annotation sessions, with the later session labelling 70% POSITIVE against 45% in the earlier one. Treat the kappa figure as a joint statement about both label sources, not about the machine alone.

The test split contains no `weak_model` labels by design.

### Ticker Attribution

`src/repair_tickers.py` scores every candidate ticker in an article using three signals in priority order: tickers shown in the page's own stock widget, tickers named explicitly in the text, and tickers inferred from full company names. The highest-scoring candidate becomes `primary_ticker`, with the score gap kept in `pt_margin`. Manual audit of 30 samples put accuracy at roughly 85%.

### Personal and Sensitive Information

All content is published financial news. Named individuals appear only in their public professional capacity as company officers or regulators. No private personal data.

### Limitations and Bias

1. Single source (CafeF); outlet-specific framing is baked in.
2. 4,119 of 5,397 labels come from a weak propagation model.
3. Machine-vs-human kappa is only 0.5505; part of that is human annotation drift.
4. Ticker attribution accuracy ~85%; wrong ticker means the label describes the wrong firm.
5. Short window (late 2024 to mid 2026), less than one full market cycle.
6. Class imbalance: NEUTRAL dominates, NEGATIVE is the rarest class.
7. Headline and lead only, no full-text reasoning.
8. Event clustering: firms in the same sector often get news on the same day.

### Citation

```bibtex
@misc{vnfinsentiment,
  title  = {vn-fin-sentiment: Ticker-level sentiment for Vietnamese financial news},
  author = {<Your Name>},
  year   = {2026},
  url    = {https://huggingface.co/datasets/<username>/vn-fin-sentiment}
}
```

---

## Tiếng Việt

### Tóm tắt

5,397 tin tài chính tiếng Việt từ CafeF, mỗi tin gắn với đúng một mã cổ phiếu và một nhãn POSITIVE / NEUTRAL / NEGATIVE. Nhãn đứng từ góc nhìn nhà đầu tư đang nắm giữ mã đó: tin này làm tăng hay giảm kỳ vọng giá trị doanh nghiệp? Các split chia thuần theo thời gian để tránh rò rỉ thông tin tương lai.

### Nhiệm vụ hỗ trợ

Phân loại văn bản 3 lớp. Cũng dùng được làm đầu vào cho event study, mỗi bài là một tín hiệu có ngày tháng gắn với một mã.

### Ngôn ngữ

Tiếng Việt (`vi`).

### Mô tả cột

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `id` | string | Định danh bài viết |
| `datetime` | timestamp | Thời điểm đăng |
| `date` | string | Ngày đăng, `YYYY-MM-DD` |
| `primary_ticker` | string | Mã cổ phiếu mà nhãn nói tới |
| `ticker_in_head` | bool | Mã có xuất hiện trong tiêu đề hoặc sapo không |
| `title` | string | Tiêu đề |
| `sapo` | string | Đoạn dẫn |
| `text` | string | Nội dung bài |
| `text_input` | string | Đầu vào mô hình, dạng `"[MÃ] tiêu đề. sapo"` |
| `label` | string | POSITIVE / NEUTRAL / NEGATIVE |
| `label_source` | string | `human` hoặc `llm` |
| `source_detail` | string | `human`, `claude_manual`, `weak_model` |
| `confidence_val` | float | Độ tin cậy nhãn, bằng 1.0 với nhãn người |
| `split` | string | train / validation / test |

### Chia split

| split   |    n |   NEGATIVE |   NEUTRAL |   POSITIVE | từ         | đến        |
|:--------|-----:|-----------:|----------:|-----------:|:-----------|:-----------|
| train   | 3777 |        389 |      2337 |       1051 | 2024-12-16 | 2026-04-21 |
| val     |  810 |         92 |       529 |        189 | 2026-04-21 | 2026-06-19 |
| test    |  810 |        140 |       409 |        261 | 2026-06-19 | 2026-08-18 |
| FULL    | 5397 |        621 |      3275 |       1501 | 2024-12-16 | 2026-08-18 |

### Vì sao xây dataset này

NLP tài chính tiếng Việt thiếu một bộ dữ liệu sentiment công khai vừa gắn nhãn theo từng mã cổ phiếu cụ thể thay vì tâm lý thị trường chung, vừa chia theo thời gian để event study phía sau không bị nhiễm rò rỉ tương lai. Dataset này nhắm vào cả hai khoảng trống đó.

### Nguồn dữ liệu

Thu thập từ CafeF (cafef.vn), khoảng 2024-12-16 đến 2026-08-18. Chỉ phát hành lại tiêu đề, đoạn dẫn và metadata.

### Quy trình gán nhãn

Ba mức, ghi rõ ở cột `source_detail`:

| Mức | Số bài | Cách gán |
|---|---:|---|
| `human` | 143 | Người đọc và gán tay |
| `claude_manual` | 1,135 | Trợ lý AI đọc từng bài, bám labeling guide |
| `weak_model` | 4,119 | Mô hình TF-IDF + từ điển + hồi quy logistic lan truyền |

Cohen's kappa giữa nhãn máy và nhãn người: **0.5505**, dưới ngưỡng 0,60. Chẩn đoán chi tiết ở `docs/prelabel_local_report.md`: chính tập nhãn người bị trôi tiêu chí giữa hai phiên gán, phiên sau có 70% POSITIVE trong khi phiên trước là 45%. Con số kappa này nói về cả hai nguồn nhãn, không riêng gì máy.

Test split được thiết kế để không chứa nhãn `weak_model` nào.

### Gán mã cổ phiếu

`src/repair_tickers.py` chấm điểm mọi mã ứng viên trong bài theo ba tín hiệu xếp thứ tự ưu tiên: mã hiện trên widget cổ phiếu của trang, mã được nhắc đích danh trong bài, mã suy ra từ tên đầy đủ doanh nghiệp. Mã điểm cao nhất thành `primary_ticker`, khoảng cách điểm lưu ở `pt_margin`. Audit tay 30 mẫu cho độ chính xác khoảng 85%.

### Thông tin cá nhân

Toàn bộ là tin tài chính đã công bố. Cá nhân được nhắc tên chỉ với tư cách công khai là lãnh đạo doanh nghiệp hoặc cơ quan quản lý. Không chứa dữ liệu cá nhân riêng tư.

### Hạn chế và thiên lệch

1. Một nguồn duy nhất (CafeF), mang theo cách đưa tin riêng của tòa soạn đó.
2. 4,119/5,397 nhãn do mô hình yếu lan truyền.
3. Kappa máy vs người mới 0.5505, một phần do người gán trôi tiêu chí.
4. Độ chính xác gán mã ~85%; sai mã thì nhãn mô tả nhầm doanh nghiệp.
5. Khoảng thời gian ngắn, chưa đủ một chu kỳ thị trường.
6. Mất cân bằng lớp: NEUTRAL áp đảo, NEGATIVE hiếm nhất.
7. Chỉ tiêu đề và đoạn dẫn, không suy luận trên toàn văn.
8. Chồng lấn sự kiện: nhiều mã cùng ngành thường có tin cùng ngày.

