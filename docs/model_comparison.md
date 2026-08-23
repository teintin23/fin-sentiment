# So sánh mô hình — Baseline vs PhoBERT

Cùng một test set 810 bài, cùng seed 42. Dự đoán PhoBERT đọc từ `models/phobert-vnfin/test_predictions.json`.

## 1. Bảng so sánh chính

### (a) Toàn bộ test

| model              |   n |   macro_F1 |   accuracy |   weighted_F1 |   F1_NEGATIVE |   F1_NEUTRAL |   F1_POSITIVE |
|:-------------------|----:|-----------:|-----------:|--------------:|--------------:|-------------:|--------------:|
| Baseline TF-IDF+LR | 810 |     0.7483 |     0.7679 |        0.7659 |        0.7016 |       0.8066 |        0.7366 |
| PhoBERT            | 810 |     0.7948 |     0.8    |        0.8002 |        0.7855 |       0.8168 |        0.7821 |

### (b) Chỉ phần nhãn người (n=22)

| model              |   n |   macro_F1 |   accuracy |   weighted_F1 |   F1_NEGATIVE |   F1_NEUTRAL |   F1_POSITIVE |
|:-------------------|----:|-----------:|-----------:|--------------:|--------------:|-------------:|--------------:|
| Baseline TF-IDF+LR |  22 |     0.5165 |     0.5    |        0.5708 |        0.7273 |       0.2667 |        0.5556 |
| PhoBERT            |  22 |     0.5225 |     0.5455 |        0.612  |        0.7692 |       0.1667 |        0.6316 |

> Cỡ mẫu 22 bài quá nhỏ, khoảng tin cậy rộng tới mức bảng (b) không kết luận được gì chắc chắn. Giữ lại cho đủ, không dùng để so hơn kém.

## 2. Kiểm định McNemar

| | PhoBERT đúng | PhoBERT sai |
|---|---:|---:|
| **Baseline đúng** | 584 | 38 |
| **Baseline sai** | 64 | 124 |

- Phương pháp: chi-square hieu chinh lien tuc, stat=6.127
- p-value = **1.331e-02**
- alpha = 0.05

**Kết luận:** bác bỏ giả thuyết hai mô hình sai như nhau. PhoBERT sửa đúng 64 bài mà baseline sai, đổi lại làm hỏng 38 bài baseline đã đúng. Chênh lệch này có ý nghĩa thống kê.

## 3. Phân tích lỗi

### Nhóm 1 — cả hai cùng sai (124 bài)

| Ticker | Tiêu đề | Thật | Baseline | PhoBERT |
|---|---|:--:|:--:|:--:|
| VHM | Vinhomes được giao đất để triển khai dự án 6.000 tỷ tại tỉnh có diện tích nhỏ nhất Việt  | POSITIVE | NEUTRAL | NEUTRAL |
| NVL | Novaland lên kế hoạch phát hành tổng cộng hơn 1,7 tỷ cổ phiếu | NEGATIVE | NEUTRAL | NEUTRAL |
| VPL | Một doanh nghiệp họ Vingroup muốn bán cổ phiếu ưu đãi giá 80.000 đồng | NEUTRAL | POSITIVE | NEGATIVE |
| BAX | TP.HCM miễn hơn 7.000 tỷ phí hạ tầng cảng biển, hỗ trợ doanh nghiệp trong 3 năm | POSITIVE | NEUTRAL | NEUTRAL |
| NVL | Novaland tuyển hơn 1.000 nhân sự xây dựng, chuẩn bị cho chu kỳ phát triển mới | POSITIVE | NEUTRAL | NEUTRAL |
| MSR | Masan High-Tech Materials (MSR) đón tin vui tại "mỏ vàng" 115 triệu tấn vonfram đa kim | POSITIVE | NEUTRAL | NEUTRAL |
| FPT | Một doanh nghiệp “họ” FPT lên kế hoạch trả cổ tức tiền mặt tỷ lệ 100% | NEUTRAL | POSITIVE | POSITIVE |
| KSF | Khởi công đường N1 và D7 bao quanh Noble Crystal Riverside: Gỡ nút thắt hạ tầng quan trọ | POSITIVE | NEUTRAL | NEUTRAL |
| VJC | Vietjet thay đổi thành viên Uỷ ban Kiểm toán | NEUTRAL | POSITIVE | POSITIVE |
| VJC | Vietjet tăng vốn lên gần 7.700 tỷ đồng sau đợt trả cổ tức | NEUTRAL | POSITIVE | POSITIVE |
| VIC | Ngân hàng Nhà nước áp dụng cơ chế đặc biệt cho 18 dự án của Vingroup, Sun Group và Maste | POSITIVE | NEUTRAL | NEUTRAL |
| YEG | Cổ phiếu nhà sản xuất “Anh trai vượt ngàn chông gai” bất ngờ cháy hàng trước giờ G | POSITIVE | NEUTRAL | NEUTRAL |
| NRC | Nữ doanh nhân sinh năm 1994 là vợ ông Đỗ Thành Nhân được bầu vào HĐQT một công ty bất độ | NEGATIVE | POSITIVE | NEUTRAL |
| GEL | ĐHĐCĐ thường niên 2026 Hạ tầng GELEX (GEL): Định hình giai đoạn phát triển mới | NEUTRAL | POSITIVE | POSITIVE |
| VHM | Vingroup chuyển quyền sở hữu hơn 15,2 triệu cổ phiếu Vinhomes | NEUTRAL | NEGATIVE | POSITIVE |

Đây là phần khó nhất. Đọc qua thì thấy chủ yếu rơi vào ba dạng: tin vĩ mô hoặc tin ngành chỉ nhắc tên doanh nghiệp thoáng qua; tin có tín hiệu ngược chiều trong cùng một câu (doanh thu tăng nhưng lợi nhuận giảm); và tin mà `primary_ticker` gán sai nên nhãn đúng cũng không liên quan tới nội dung.

### Nhóm 2 — PhoBERT đúng, baseline sai (64 bài)

| Ticker | Tiêu đề | Thật | Baseline | PhoBERT |
|---|---|:--:|:--:|:--:|
| QCG | Ông Nguyễn Quốc Cường: Giá cổ phiếu QCG đang thấp hơn nhiều so với giá trị sổ sách | NEUTRAL | NEGATIVE | NEUTRAL |
| VIC | Chi tiết hướng tuyến 5 dự án metro Hà Nội do Vingroup làm tổng thầu | POSITIVE | NEUTRAL | POSITIVE |
| CTG | Việt Nam có ngân hàng thứ hai đạt mốc tài sản 3 triệu tỷ đồng | POSITIVE | NEUTRAL | POSITIVE |
| DIG | Xử phạt DIG, buộc hoàn tiền cho nhà đầu tư mua trái phiếu | NEGATIVE | NEUTRAL | NEGATIVE |
| PVD | Sở hữu “hạm đội” chuyên đào kho báu ngoài khơi, DN được dự báo lãi tăng hơn 43%, hưởng l | POSITIVE | NEUTRAL | POSITIVE |
| VPL | Vinpearl (VPL) vừa hút thành công 255 triệu USD từ các quỹ đầu tư quốc tế | POSITIVE | NEUTRAL | POSITIVE |
| LPB | Chi 7.555 tỷ mua cổ phiếu Lộc Phát, nhóm nhà đầu tư tạm lãi hơn 1.500 tỷ khi cổ phiếu cò | POSITIVE | NEGATIVE | POSITIVE |
| DGC | Hóa chất Đức Giang dự kiến thời gian khắc phục ý kiến ngoại trừ của kiểm toán về hàng tồ | NEUTRAL | NEGATIVE | NEUTRAL |
| ABB | Dự án chung cư “hot” bậc nhất Hà Nội, từng chốt 1.000 căn hộ chỉ sau 10 ngày mở bán, đan | NEUTRAL | POSITIVE | NEUTRAL |
| ACV | “Trái tim” trị giá 35.000 tỷ đồng tại sân bay lớn nhất Việt Nam đang được 7.300 công nhâ | POSITIVE | NEUTRAL | POSITIVE |

Phần lớn là câu mà nghĩa phụ thuộc trật tự từ và phủ định, thứ TF-IDF không nắm được vì nó chỉ đếm n-gram rời rạc.

### Nhóm 3 — baseline đúng, PhoBERT sai (38 bài)

| Ticker | Tiêu đề | Thật | Baseline | PhoBERT |
|---|---|:--:|:--:|:--:|
| PSC | Petrovietnam ký loạt hợp đồng dầu khí, bổ sung nguồn khí cho phát điện từ năm 2027 | NEUTRAL | NEUTRAL | POSITIVE |
| VIC | Những dự án nào của Vingroup, Sun Group và Masterise được loại trừ dư nợ khi tính room t | POSITIVE | POSITIVE | NEUTRAL |
| HVN | 'Đại bàng' Mỹ bảo lãnh hơn 2,9 tỷ USD cho Vietnam Airlines mua 50 máy bay | POSITIVE | POSITIVE | NEUTRAL |
| MWG | MWG phát hành hơn 7 triệu cổ phiếu ESOP giá 10.000 đồng, ông Nguyễn Đức Tài không góp mặ | NEUTRAL | NEUTRAL | POSITIVE |
| NKG | Thép Nam Kim tăng vốn lên hơn 4.900 tỷ đồng sau đợt trả cổ tức | NEUTRAL | NEUTRAL | POSITIVE |
| HVN | Vietnam Airlines lãi lớn 7.600 tỷ nhưng hẹn sau 2032 mới hết lỗ lũy kế để chia cổ tức | POSITIVE | POSITIVE | NEGATIVE |
| MBS | MBS dự báo lợi nhuận một "đại gia" dầu khí có thể tăng trưởng hơn 380% trong quý 2 | NEUTRAL | NEUTRAL | POSITIVE |
| VPB | VPBank đứng đầu nhóm ngân hàng tư nhân Việt Nam trong bảng xếp hạng Forbes Global 2000 | NEUTRAL | NEUTRAL | POSITIVE |
| VCB | Lợi nhuận một ngân hàng được dự báo vượt Vietcombank, dẫn đầu toàn ngành trong quý II/20 | NEUTRAL | NEUTRAL | POSITIVE |
| HVN | Chuyến bay không có bất kỳ hành khách nào cất cánh đã trở về Nội Bài, lập kỳ tích bay dà | NEUTRAL | NEUTRAL | POSITIVE |

Thường là tin có từ khóa rất mạnh và rõ ràng, đúng thế mạnh của mô hình đếm từ, còn PhoBERT thì bị ngữ cảnh xung quanh kéo lệch.

## 4. Nhận xét

- PhoBERT hơn baseline **+0.0465** macro-F1 (0.7483 → 0.7948).
- Mức tăng lớn nhất nằm ở lớp NEGATIVE: 0.702 → 0.785. Đây là lớp ít mẫu nhất và cũng là lớp quan trọng nhất cho event study.
- Còn 124 bài cả hai cùng sai, chiếm 15.3% test set. Phần này khó có thể cải thiện bằng đổi kiến trúc, gốc rễ nằm ở chất lượng nhãn train và độ chính xác gán mã cổ phiếu.
- Train chứa ~89% nhãn `weak_model`. Cả hai mô hình đều đang học từ nhãn nhiễu, nên con số này là sàn chứ không phải trần.
