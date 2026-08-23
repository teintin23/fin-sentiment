# Báo cáo gán nhãn cục bộ (prelabel_local)

Thay thế cho `prelabel_llm.py`. Không gọi API, không tốn phí.

## 1. Quy mô

- Tổng số bài cần nhãn: **5,549**
- Nhãn hạt giống gán tay: **1,170** (21.1%)
- Nhãn người trong gold seed (không dùng để huấn luyện): 143

## 2. Chất lượng mô hình lan truyền (5-fold CV trên tập hạt giống)

- accuracy: **0.7709**
- macro-F1: **0.7481**

### Ma trận nhầm lẫn

| Claude gan tay   |   NEGATIVE |   NEUTRAL |   POSITIVE |
|:-----------------|-----------:|----------:|-----------:|
| NEGATIVE         |        116 |        48 |         16 |
| NEUTRAL          |         21 |       526 |         66 |
| POSITIVE         |         11 |       106 |        260 |

### Độ tin cậy của confidence

| khoảng conf   |   n |   conf TB |   accuracy thực |
|:--------------|----:|----------:|----------------:|
| 0.00-0.50     | 112 |     0.455 |           0.464 |
| 0.50-0.60     | 196 |     0.553 |           0.566 |
| 0.60-0.70     | 254 |     0.653 |           0.76  |
| 0.70-0.80     | 288 |     0.751 |           0.868 |
| 0.80-0.90     | 260 |     0.846 |           0.915 |
| 0.90-1.01     |  60 |     0.927 |           0.967 |

Cột `accuracy thực` cho biết trong nhóm dự đoán có confidence rơi vào
khoảng đó, thực tế bao nhiêu phần trăm đúng. Dùng bảng này để chọn
ngưỡng lọc ở `finalize_dataset.py`.

## 3. Cách đọc cột label_source trong labeled_llm.jsonl

| Giá trị | Nghĩa | confidence |
|---|---|---|
| `claude_manual` | Trợ lý đọc từng bài và gán theo labeling_guide | 0.95 |
| `weak_model` | Mô hình lan truyền dự đoán | xác suất đã hiệu chuẩn |

## 4. Hạn chế

- Nhãn hạt giống do một mô hình ngôn ngữ gán, không phải chuyên gia tài chính.
- Mô hình lan truyền chỉ nhìn tiêu đề + sapo, không đọc toàn văn.
- Tập hạt giống lệch về nửa cuối giai đoạn dữ liệu, nên phần train
  (thời gian sớm hơn) chủ yếu là nhãn `weak_model`, độ nhiễu cao hơn.
- Nhãn `weak_model` chỉ nên coi là nhãn yếu. Mọi kết luận về chất lượng
  mô hình ở P2 phải báo cáo riêng trên phần `label_source == "human"`.

---

## 5. Đối chiếu với nhãn người (kết quả chạy thực tế)

Chạy `python src/eval_labels.py` trên 143 nhãn người hiện có:

| Chỉ số | Giá trị |
|---|---|
| N khớp | 143 |
| Accuracy | 0.713 |
| **Cohen's kappa** | **0.5505** |
| Macro-F1 | 0.712 |

Kappa 0.55 nằm ở mức "trung bình", dưới ngưỡng 0.60 mà PROMPTS.md đặt ra làm
điểm dừng. Nhưng con số tổng gộp này che mất điều quan trọng nhất.

### 5.1 Tách theo nguồn nhãn máy

| Nguồn | n | Accuracy | Kappa |
|---|---:|---:|---:|
| `claude_manual` (đọc từng bài) | 35 | 0.829 | **0.729** |
| `weak_model` (lan truyền) | 108 | 0.676 | 0.480 |

### 5.2 Tách theo cột `reused` của gold seed

| Nhóm gold | n | Accuracy | Kappa | % POSITIVE do người gán |
|---|---:|---:|---:|---:|
| `reused=True` — nhãn từ phiên cũ | 82 | 0.805 | **0.693** | 45% |
| `reused=False` — nhãn phiên mới | 61 | 0.590 | **0.363** | **70%** |

Đây mới là nguyên nhân chính. Hai phiên gán nhãn của cùng một người cho ra hai
tiêu chí khác nhau, và phiên mới lệch mạnh về POSITIVE.

### 5.3 Bằng chứng bổ trợ

- 26/41 ca bất đồng (63%) có dạng **người=POSITIVE, máy=NEUTRAL**. Bất đồng tập
  trung vào đúng một đường ranh giới, không rải đều — dấu hiệu lệch tiêu chí
  hệ thống chứ không phải nhiễu ngẫu nhiên.
- Các tiêu đề trong nhóm đó phần lớn là tin CSR, triển lãm, bổ nhiệm nhân sự,
  tin vĩ mô có nhắc tên doanh nghiệp. Ví dụ: *"100.000 m2 tôn được gửi khẩn cấp
  tới người dân Đắk Lắk"*, *"một nữ lãnh đạo LPBank tham gia ban điều hành
  Sacombank"*, *"Gỗ An Cường hiện diện tại Triển lãm Rạng Rỡ Việt Nam"*.
  Theo `docs/labeling_guide.md` mục 3 và mục 6 thì đây là NEUTRAL.
- Phân bố nhãn người: 55,9% POSITIVE / 25,2% NEUTRAL / 18,9% NEGATIVE.
  Một corpus tin tài chính có hơn một nửa là tin tốt là điều khó xảy ra.
- Nếu chỉ xét bài toán nhị phân "có phải tin xấu không", kappa đạt 0.661 —
  tức là phần NEGATIVE, phần quan trọng nhất cho event study ở P3, vẫn ổn.

### 5.4 Việc cần làm trước khi sang P1.4

1. Chạy `python src/audit_gold.py` → sinh `data/interim/gold_recheck.csv`
   gồm 41 dòng bất đồng, xếp theo mức đáng ngờ giảm dần (25 dòng thuộc nhãn mới,
   16 dòng thuộc nhãn cũ).
2. Đọc lại 41 dòng đó, bám sát mục 6 của labeling_guide: *phân vân giữa NEUTRAL
   và một cực thì chọn NEUTRAL*. Điền cột `label_final`, để trống nghĩa là giữ nguyên.
3. `python src/audit_gold.py --apply gold_recheck.csv` rồi `python src/eval_labels.py`.
4. Gán nốt 57 dòng còn trống bằng `src/label_tool.py`. Sau mỗi 20 bài nên dừng
   nhìn lại phân bố; nếu POSITIVE vượt quá khoảng 40% thì gần như chắc chắn đang
   trôi tiêu chí.

Nếu sau bước này kappa vượt 0.60 thì đi tiếp P1.4. Nếu vẫn không, vấn đề nằm ở
guideline chứ không nằm ở phương pháp gán nhãn.
