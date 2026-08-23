# Báo cáo Chất lượng Nhãn LLM

**Generated:** 2026-08-23 13:04:01  
**Gold set:** `data/interim/gold_seed.csv` (143 bài khớp)  
**Nhãn máy:** `data/interim/labeled_auto.jsonl`

---

## Chỉ số tổng hợp

| Chỉ số | Giá trị |
|--------|--------:|
| N (matched) | 143 |
| Accuracy | 0.7133 (71.3%) |
| **Cohen's kappa** | **0.5505** |
| Macro-F1 | 0.7118 |

## Chỉ số theo nhãn

| Nhãn | Precision | Recall | F1 |
|------|----------:|-------:|---:|
| POSITIVE | 0.893 | 0.625 | 0.735 |
| NEUTRAL | 0.531 | 0.944 | 0.680 |
| NEGATIVE | 0.783 | 0.667 | 0.720 |

## Confusion Matrix

> Hàng = nhãn người (Human), Cột = nhãn máy

| Human \ LLM | **POSITIVE** | **NEUTRAL** | **NEGATIVE** |
|-------------|------:|------:|------:|
| **POSITIVE** | 50 | 26 | 4 |
| **NEUTRAL** | 1 | 34 | 1 |
| **NEGATIVE** | 5 | 4 | 18 |

## Phân bố nhãn

| Nhãn | Người (%) | LLM (%) |
|------|----------:|--------:|
| POSITIVE | 55.9% | 39.2% |
| NEUTRAL | 25.2% | 44.8% |
| NEGATIVE | 18.9% | 16.1% |

## Bất đồng chi tiết

Tổng số bất đồng: **41** / 143 bài (28.7%)

### Nhóm bất đồng

| Human → LLM | Số bài | Tỷ lệ |
|-------------|-------:|------:|
| Human=POSITIVE vs LLM=NEUTRAL | 26 | 18.2% |
| Human=NEGATIVE vs LLM=POSITIVE | 5 | 3.5% |
| Human=NEGATIVE vs LLM=NEUTRAL | 4 | 2.8% |
| Human=POSITIVE vs LLM=NEGATIVE | 4 | 2.8% |
| Human=NEUTRAL vs LLM=NEGATIVE | 1 | 0.7% |
| Human=NEUTRAL vs LLM=POSITIVE | 1 | 0.7% |

### Chi tiết từng case bất đồng

**[1]** `HPG` — 100.000 m2 tôn được gửi khẩn cấp tới người dân Đắk Lắk để sửa nhà sau mưa lũ  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.82)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[2]** `UDJ` — Đà Nẵng tính toán di dời gần 400.000 ngôi mộ để tái thiết đô thị, phát triển hạ tầng ven biển  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.83)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[3]** `STB` — Cùng với ông Nguyễn Đức Thụy, một nữ lãnh đạo của LPBank chính thức tham gia ban điều hành Sacombank  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.96)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[4]** `VIC` — VinFast đổ bộ Congo: Tham vọng bán 100.000 xe máy ngang mức đỉnh tại Việt Nam chỉ trong 1 quý, chính sách 'khai tử' xe cũ có 'đỡ' lại cơn khát điện năng?  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.82)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[5]** `VNG` — Một công ty AI cùng hệ sinh thái với Zalo vừa nhận giấy chứng nhận Công nghệ cao đầu tiên của TP.HCM  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.68)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[6]** `ACG` — Gỗ An Cường hiện diện ấn tượng tại Triển lãm "Rạng Rỡ Việt Nam"  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.95)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[7]** `DGC` — Hóa chất Đức Giang gặp biến cố, ai sẽ hưởng lợi?  
- Người: `NEGATIVE` | LLM: `POSITIVE` (conf=0.77)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[8]** `MSN` — SK thoái hết vốn khỏi Masan, cổ phiếu tăng đột biến: Điều gì sẽ xảy ra?  
- Người: `NEUTRAL` | LLM: `NEGATIVE` (conf=0.59)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[9]** `VIC` — Vingroup của tỷ phú Phạm Nhật Vượng đã lấy ý kiến người dân ở một xã Hà Nội về KĐT thể thao Olympic có sân vận động 135.000 chỗ, lớn nhất thế giới  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.54)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[10]** `TCX` — TCBS sẽ phát hành hơn 462 triệu cổ phiếu trả cổ tức  
- Người: `NEGATIVE` | LLM: `NEUTRAL` (conf=0.66)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[11]** `CAN` — Ghi nhận tình hình thu hồi đồ hộp Hạ Long: Nhiều cửa hàng Bách Hóa Xanh và WinMart+ sẵn sàng hoàn tiền cho khách dù không còn hóa đơn  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.79)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[12]** `TCB` — Lộ diện 'đại gia' gửi hàng trăm tỷ đồng không kỳ hạn  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.95)  
- LLM reason: *gan tay theo docs/labeling_guide.md*

**[13]** `PGB` — Thành viên HĐQT PGBank thay đổi kế hoạch mua cổ phiếu  
- Người: `NEGATIVE` | LLM: `POSITIVE` (conf=0.48)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[14]** `MWG` — Lãnh đạo MWG nói gì về mục tiêu IPO Bách Hóa Xanh?  
- Người: `POSITIVE` | LLM: `NEGATIVE` (conf=0.47)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[15]** `ACB` — Một ngân hàng vừa giải ngân 700 tỉ đồng cho người trẻ vay mua nhà  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.94)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[16]** `VIC` — Loạt “ông lớn” Vingroup, Sun Group, BRG, Mường Thanh… “đổ bộ” về tỉnh biên giới giáp Trung Quốc và Lào  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.60)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[17]** `SSB` — Khu du lịch 6.000 tỷ đồng từng lớn nhất Đông Nam Á của vợ chồng bà Nguyễn Phương Hằng tung chương trình đặc biệt trong duy nhất 1 ngày  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.95)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[18]** `KBC` — Lộ diện 9 nhà đầu tư bỏ ra hơn 4.100 tỷ đồng mua cổ phiếu riêng lẻ của Kinh Bắc (KBC), VPBankS mua "nhiệt tình" nhất  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.54)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[19]** `VJC` — Vietjet cất nóc hangar chuẩn quốc tế, khai trương chuyến bay tới Cảng Hàng không quốc tế Long Thành  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.65)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[20]** `ABB` — ABBank kích hoạt 3 động lực tăng trưởng mới  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.55)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[21]** `BAX` — TP.HCM miễn hơn 7.000 tỷ phí hạ tầng cảng biển, hỗ trợ doanh nghiệp trong 3 năm  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.95)  
- LLM reason: *gan tay theo docs/labeling_guide.md*

**[22]** `VCG` — Vinaconex bất ngờ muốn bán hơn 18% vốn công ty có cổ phiếu tăng nóng  
- Người: `NEGATIVE` | LLM: `POSITIVE` (conf=0.87)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[23]** `HAX` — Haxaco lên kế hoạch lợi nhuận gấp 5 lần  
- Người: `POSITIVE` | LLM: `NEGATIVE` (conf=0.56)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[24]** `KPF` — “Cú bắt tay” thế kỷ cùng KPF và khát vọng đưa Bãi Cháy vươn tầm quốc tế  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.95)  
- LLM reason: *gan tay theo docs/labeling_guide.md*

**[25]** `TPB` — TPBank Biz Expense: Lời giải mới cho bài toán quản trị chi phí doanh nghiệp  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.95)  
- LLM reason: *gan tay theo docs/labeling_guide.md*

**[26]** `PVT` — PVT Logistics sắp phát hành hơn 13 triệu cổ phiếu trả cổ tức năm 2024  
- Người: `NEGATIVE` | LLM: `POSITIVE` (conf=0.63)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[27]** `HVN` — Loại “tiền tệ” có giá trị phát hành 300 tỷ USD/năm, quyền lực thứ 3 thế giới sau USD, Euro: Vietnam Airlines đang quyết liệt khai thác  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.50)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[28]** `MBS` — MBS dự báo lợi nhuận quý 2 của 13 ngân hàng: Hai nhà băng cùng tăng trưởng trên 45%, VietinBank lãi hơn 11.000 tỷ đồng  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.95)  
- LLM reason: *gan tay theo docs/labeling_guide.md*

**[29]** `TDH` — Cổ phiếu Thuduc House (TDH) tăng kịch trần sau tin thắng kiện vụ án hoàn thuế 365 tỷ đồng  
- Người: `POSITIVE` | LLM: `NEGATIVE` (conf=0.45)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[30]** `VPL` — Vinpearl ra mắt Vinpearl Legendlux – Thương hiệu khách sạn siêu sang tôn vinh bản sắc Việt  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.82)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[31]** `SGT` — Công ty ông Đặng Thành Tâm dồn vốn làm data center và máy bay không người lái  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.76)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[32]** `FPT` — FPT ảnh hưởng thế nào khi thay đổi cổ đông nắm 50,17% cổ phần FPT Telecom?  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.66)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[33]** `CTD` — Coteccons (CTD) đặt cược vào mục tiêu 30.000 tỷ: Tham vọng vượt đỉnh lịch sử đi kèm nỗi lo dòng tiền âm hơn 1.100 tỷ đồng.  
- Người: `NEGATIVE` | LLM: `POSITIVE` (conf=0.94)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[34]** `TPB` — TPBank tái hiện từng bước xanh hóa qua Báo cáo Phát triển bền vững  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.93)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[35]** `QCG` — Quyết định quan trọng với Quốc Cường Gia Lai  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.55)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[36]** `TTD` — Cổ phiếu bệnh viện chia thưởng 100%, nhà đầu tư lãi gần gấp đôi trong 1 tháng  
- Người: `NEUTRAL` | LLM: `POSITIVE` (conf=0.95)  
- LLM reason: *gan tay theo docs/labeling_guide.md*

**[37]** `CAN` — NÓI THẲNG: Patê 'bẩn' của Hạ Long Canfoco và những người không thể vô can  
- Người: `NEGATIVE` | LLM: `NEUTRAL` (conf=0.81)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[38]** `DSC` — Thành viên HĐQT và cổ đông lớn không muốn tham gia đợt chào bán cổ phiếu của DSC  
- Người: `NEGATIVE` | LLM: `NEUTRAL` (conf=0.65)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[39]** `VCB` — Đẳng cấp như Vietcombank: Tất cả các nhóm nợ xấu đều giảm rất mạnh  
- Người: `POSITIVE` | LLM: `NEUTRAL` (conf=0.78)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[40]** `QCG` — Ngược dòng, cổ phiếu QCG “tím lịm” phiên thứ 2 giữa lúc VN-Index “lao dốc” mạnh  
- Người: `POSITIVE` | LLM: `NEGATIVE` (conf=0.98)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

**[41]** `HSG` — Nóng: Tập đoàn Hoa Sen lên tiếng về vụ điều tra chống bán phá giá tôn thép mạ tại Úc với biên độ lên tới 56%  
- Người: `NEGATIVE` | LLM: `NEUTRAL` (conf=0.82)  
- LLM reason: *lan truyen tu nhan hat giong (tfidf+lexicon+LR)*

## Kết quả kiểm tra

**❌ FAIL**

- Cohen's kappa = **0.5505** (ngưỡng >= 0.6)

## Đề xuất cải thiện Guideline

> Nhóm bất đồng lớn nhất: **Human=POSITIVE vs LLM=NEUTRAL** (26 bài)

- LLM NEUTRAL khi người gán POSITIVE.
- Nguyên nhân thường: tin có dấu hiệu POSITIVE nhưng thiếu con số cụ thể.
- Đề xuất: Thêm ví dụ 'POSITIVE không cần số tuyệt đối' vào Section 7 guideline.
- Xem xét hạ ngưỡng cụ thể hóa (specificity threshold) cho mảng ký hợp đồng/IPO..

> **Lưu ý:** Không tự sửa guideline — cần review thủ công trước khi cập nhật.
