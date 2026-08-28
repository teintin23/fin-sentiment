# vn-fin-sentiment: Tóm tắt nghiên cứu / Research Summary

*Tài liệu 4 trang cho người đọc lần đầu. Mọi con số sinh từ code trong repo, chạy lại được bằng các lệnh ở README.*

## Abstract (EN)

We build a Vietnamese financial news sentiment dataset of 5,397 CafeF articles (Dec 2024 - Aug 2026), labeled from the perspective of an investor holding the primary ticker mentioned. Labels come in three provenance tiers (human, AI-assisted manual, weak-model propagation); the test split is strictly chronological and fully hand-labeled. A fine-tuned PhoBERT reaches 0.795 macro-F1 versus 0.748 for a TF-IDF baseline (McNemar p = 0.013). A Brown-Warner event study on 3,616 events finds a +0.97% announcement-day abnormal-return spread between positive and negative news. We document endogeneity directly — positive articles show +0.84% pre-event drift (p = 7.4e-05), i.e. journalists write good news about rising stocks — and show the announcement-day effect survives regression controls for pre-event momentum (+0.91%, p = 3.5e-07; +1.33% on non-overlapping events, p = 0.0002). Post-announcement drift CAR[1,5] is insignificant across all samples: the labels carry information that is priced within the announcement session, and provide no tradable forecast.

## 1. Câu hỏi nghiên cứu

1. Nhãn sentiment tin tài chính tiếng Việt, định nghĩa theo góc nhìn người nắm giữ mã, có học được bằng mô hình không?
2. Nhãn đó có tương quan với lợi suất bất thường quanh ngày đăng tin không, và tương quan đó có phải chỉ là momentum trước sự kiện?
3. Nhãn có dự báo được lợi suất SAU phiên tin ra không?

## 2. Dữ liệu

5,397 bài CafeF, 12/2024 - 08/2026, 195 mã. Mỗi bài gán một `primary_ticker` bằng chấm điểm (không lấy mã đầu tiên - cách đó sai ~1/3 số bài; độ chính xác gán mã ước tính 85% qua audit tay). Input mô hình dạng `[HPG] tiêu đề + sapo` vì nhãn không xác định nếu thiếu góc nhìn mã.

Ba tầng nguồn nhãn, giữ tách bạch trong cột `label_source`:

| tầng | n | độ tin cậy |
|---|---|---|
| `human` | 143 | cao nhất |
| `claude_manual` | 1,135 | đọc từng bài theo labeling guide |
| `weak_model` | 4,119 | lan truyền từ tập hạt giống, nhiễu nhất |

Chia train/val/test **thuần theo thời gian** (chống rò rỉ nhìn trước tương lai). Test 810 bài, 100% nhãn đọc tay. Kappa nhãn máy vs người: 0.5505 - dưới ngưỡng tốt, nguyên nhân đã truy ra là tiêu chí gán trôi giữa hai phiên làm việc (`docs/prelabel_local_report.md`).

## 3. Mô hình

| mô hình | macro-F1 test | ghi chú |
|---|---|---|
| TF-IDF + LogReg | 0.7483 | siêu tham số chọn trên phần val nhãn tay, tránh tự chấm điểm mình |
| PhoBERT fine-tuned | **0.7948** | tách từ underthesea trước tokenize |
| lexicon rule | 0.5165 | tham chiếu dưới |

McNemar p = 0.013: khác biệt có ý nghĩa. Trần hiệu năng hiện bị chặn bởi chất lượng nhãn train (76% weak_model), không phải kiến trúc.

## 4. Event study

Khung Brown & Warner (1985), market model ước lượng trên [-130,-11], t0 điều chỉnh theo giờ đăng (sau 15:00 tính phiên sau), gộp sự kiện trùng mã-phiên, kiểm định t cắt ngang.

**Kết quả gốc.** POS-NEG CAR[0,0] = +0.972% (p = 4.3e-08, 3,616 sự kiện). Đơn điệu POS > NEUTRAL ≈ 0 > NEG ở mọi cửa sổ, bền qua market-adjusted và mẫu không chồng lấn (+1.321%, p = 0.0003, 663 sự kiện).

**Vấn đề tự phát hiện.** Giả dược [-5,-1] dương ở nhóm POSITIVE: +0.835% (p = 7.4e-05). Báo viết tin tốt về mã đang tăng. Nếu dừng ở đây, không phân biệt được "tin mang thông tin" với "tin chạy theo giá".

**Xử lý** (`docs/event_study_momentum.md`), ba kiểm định:

1. *Hồi quy kiểm soát*: CAR[0,w] ~ POS + NEG + preCAR[-5,-1], HC1 robust SE. POS-NEG tại [0,0] còn **+0.905% (p = 3.5e-07)**; mẫu không chồng lấn **+1.329% (p = 0.0002)**; chỉ nhãn đọc tay **+1.418% (p = 1.2e-05)**. Hiệu ứng đồng thời không phải artifact momentum.
2. *Tercile |preCAR| thấp* (phi tham số): spread +0.224%, cùng chiều nhưng không có ý nghĩa (n = 298/100). Thiếu lực thống kê, không mâu thuẫn với (1).
3. *Post-drift*: CAR[1,5] ~ POS + NEG + preCAR + AR[0]. POS-NEG không có ý nghĩa ở cả 3 mẫu (p = 0.12 toàn mẫu). **Không có drift sau tin.**

## 5. Kết luận

Nhãn sentiment mang thông tin được thị trường định giá tại phiên tin ra, độc lập với momentum trước sự kiện. Không có bằng chứng về sức dự báo sau đó - nhất quán với thị trường hấp thụ tin trong phiên đầu. Event study ở đây là thước đo chất lượng nhãn bằng dữ liệu ngoài (external validation), không phải tín hiệu giao dịch.

## 6. Điều KHÔNG kết luận được

- Không kết luận tin CafeF dự báo giá. Bằng chứng nói ngược lại (mục 4.3).
- Không khái quát ra ngoài CafeF, ngoài giai đoạn 12/2024-08/2026, ngoài 195 mã có giá.
- macro-F1 0.795 đo trên tiêu đề + sapo, không phải toàn văn.
- Kappa 0.55 nghĩa là bản thân "ground truth" còn nhiễu; mọi con số F1 mang sai số nhãn.
- Không dùng cho quyết định đầu tư.

## 7. Hướng tiếp

Gán lại nhãn với guide đã chốt để nâng kappa; nguồn tin thứ hai (Vietstock/VnExpress) làm robustness; ablation toàn văn vs tiêu đề; kéo dài mẫu qua một chu kỳ thị trường.
