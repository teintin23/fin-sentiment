# Hướng dẫn Gán nhãn Sentiment Tin tài chính tiếng Việt

> **Phiên bản:** 1.0 — 2026-08-19  \
> **Dành cho:** Người gán nhãn (annotators)  \
> **Liên hệ:** Nhóm dữ liệu nội bộ

---

## Mục lục

1. [Định nghĩa nhiệm vụ](#1-định-nghĩa-nhiệm-vụ)
2. [Góc nhìn gán nhãn](#2-góc-nhìn-gán-nhãn)
3. [Ba nhãn và định nghĩa](#3-ba-nhãn-và-định-nghĩa)
4. [Bảng ví dụ thực tế](#4-bảng-ví-dụ-thực-tế)
5. [Trường hợp khó](#5-trường-hợp-khó)
6. [Quy tắc quyết khi phân vân](#6-quy-tắc-quyết-khi-phân-vân)
7. [Quy tắc xử lý bài nhắc nhiều mã](#7-quy-tắc-xử-lý-bài-nhắc-nhiều-mã)

---

## 1. Định nghĩa nhiệm vụ

**Mục tiêu:** Gán nhãn cảm xúc thị trường (*market sentiment*) cho các tin tài chính tiếng Việt thu thập từ báo điện tử.

| Thuộc tính | Chi tiết |
|---|---|
| **Đơn vị gán nhãn** | Một cặp `(title, sapo)` — tiêu đề + đoạn dẫn của bài báo |
| **Không đọc thêm** | KHÔNG cần đọc phần `body`; toàn bộ phán đoán dựa trên title + sapo |
| **Số lớp** | 3 lớp: `POSITIVE (1)`, `NEUTRAL (0)`, `NEGATIVE (-1)` |
| **Nguồn dữ liệu** | CafeF, VnExpress Kinh doanh, các báo tài chính tiếng Việt |

**Thời gian kỳ vọng mỗi mẫu:** 20–40 giây. Nếu một mẫu mất hơn 2 phút vẫn chưa quyết được, gán `NEUTRAL (0)` và ghi chú "khó".

---

## 2. Góc nhìn gán nhãn

> **Câu hỏi duy nhất bạn cần trả lời:**
> *"Nếu tôi đang CẦM mã cổ phiếu được nhắc tới trong bài này, tin này khiến tôi **vui** hay **lo**?"*

### Góc nhìn đúng: Nhà đầu tư đang nắm giữ

Bạn đóng vai một nhà đầu tư đã mua và đang nắm giữ cổ phiếu `primary_ticker` của bài báo.
Bạn chỉ quan tâm đến một điều: **tác động của tin này lên kỳ vọng giá trị cổ phiếu trong tương lai**.

### Những gì KHÔNG làm cơ sở gán nhãn

| ❌ Sai | ✅ Đúng |
|---|---|
| Bài viết dùng từ tích cực, hào hứng → gán POSITIVE | Nội dung thực sự báo hiệu tăng giá trị → gán POSITIVE |
| Bài viết bi quan về xã hội → gán NEGATIVE | Tin chỉ ảnh hưởng xã hội, không ảnh hưởng doanh nghiệp → gán NEUTRAL |
| Tác giả có cảm xúc rõ → theo cảm xúc tác giả | Chỉ theo tác động kỳ vọng với cổ phiếu |
| Tin xấu cho xã hội nhưng tốt cho DN → NEGATIVE | Gán POSITIVE (ví dụ: giá nguyên liệu giảm → lợi nhuận tăng) |

### Ví dụ minh hoạ ngay

- **"FPT ký hợp đồng 500 triệu USD với đối tác Nhật"** → Bạn đang cầm FPT → Vui → **POSITIVE**
- **"FPT thay đổi địa chỉ trụ sở chính"** → Không ảnh hưởng giá trị → **NEUTRAL**
- **"FPT bị phạt 50 tỷ đồng vì vi phạm thuế"** → Lo → **NEGATIVE**

---

## 3. Ba nhãn và định nghĩa

### POSITIVE (1)

**Định nghĩa:** Tin làm **tăng kỳ vọng** về giá trị hoặc dòng tiền của doanh nghiệp trong tương lai.

**Dấu hiệu nhận biết:**
- Kết quả kinh doanh tăng trưởng so với cùng kỳ (YoY > +5%)
- Ký hợp đồng lớn, mở rộng thị trường, thâm nhập thị trường mới
- Lãnh đạo nội bộ mua vào cổ phiếu của chính công ty
- Chia cổ tức tiền mặt
- Nhận đầu tư chiến lược từ đối tác uy tín, kèm con số cụ thể
- Được nâng hạng tín nhiệm, rating tích cực từ tổ chức uy tín quốc tế

---

### NEUTRAL (0)

**Định nghĩa:** Thông báo hành chính, thủ tục định kỳ, tin không rõ hướng tác động,
hoặc tin PR thuần không có nội dung tài chính thực chất.

**Dấu hiệu nhận biết:**
- Họp ĐHCĐ thường niên (không có quyết định bất thường)
- Thay đổi địa chỉ, tên công ty con, quy chế nội bộ
- Giải thưởng, chứng nhận mang tính danh hiệu (không kèm số liệu tài chính)
- Ra mắt sản phẩm/tính năng nhỏ, chưa rõ tác động doanh thu
- Tin vĩ mô không gắn với mã cụ thể
- Kết quả kinh doanh đi ngang (tăng/giảm < 5% so với cùng kỳ)
- Chia cổ phiếu thưởng (không phải tiền mặt)

---

### NEGATIVE (-1)

**Định nghĩa:** Tin làm **giảm kỳ vọng** về giá trị hoặc dòng tiền của doanh nghiệp.

**Dấu hiệu nhận biết:**
- Kết quả kinh doanh sụt giảm đáng kể so với cùng kỳ (YoY < −5%)
- Lãnh đạo nội bộ bán ra cổ phiếu của chính công ty
- Bị cơ quan quản lý cảnh báo, kiểm soát, thu hồi giấy phép
- Lãnh đạo bị khởi tố, bắt giữ, điều tra hình sự
- Bị cắt margin bởi công ty chứng khoán
- Phát hành thêm cổ phiếu pha loãng (giá phát hành thấp hơn thị giá)
- Kiện tụng lớn, thiệt hại tài sản đáng kể, mất hợp đồng quan trọng

---

## 4. Bảng ví dụ thực tế

> Các ví dụ sau lấy trực tiếp từ dataset `articles_clean.parquet`. Tiêu đề là nguyên bản từ bài báo.

| # | Tiêu đề (title) | Ticker | Nhãn | Lý do |
|---|---|:---:|:---:|---|
| 1 | VIB: Lợi nhuận quý 1.2025 đạt hơn 2.400 tỷ đồng, CASA tăng 17%, thực hiện chia cổ tức 21% | VIB | **POSITIVE** | Lợi nhuận tăng 7% YoY, CASA tăng 17%, chia cổ tức tiền mặt 21% — ba tín hiệu tích cực cùng lúc. |
| 2 | LPBank "về đích" 2025 ấn tượng, tổng tài sản vượt 605.000 tỷ đồng, lợi nhuận tăng 17% | LPB | **POSITIVE** | Lợi nhuận trước thuế tăng 17% YoY — kết quả kinh doanh rõ ràng vượt kỳ vọng. |
| 3 | Thế Giới Di Động và Điện máy Xanh tăng trưởng doanh thu 15% dù đóng hơn 150 cửa hàng | MWG | **POSITIVE** | Doanh thu tăng 14% YoY trong khi giảm số cửa hàng — hiệu quả hoạt động cải thiện rõ nét. |
| 4 | IPO Điện Máy Xanh: Dragon Capital đăng ký mua tối thiểu 50 triệu USD, khẳng định giá đang thấp đáng kể | MWG | **POSITIVE** | Quỹ lớn cam kết đầu tư và xác nhận định giá thấp — tín hiệu xác nhận giá trị nội tại. |
| 5 | OKX và VPBank ký thỏa thuận hợp tác chiến lược phát triển hợp tác trong lĩnh vực tài sản số và blockchain | VPB | **POSITIVE** | Hợp tác có góp vốn thực tế vào CAEX, mở hướng kinh doanh tài sản số — tác động doanh thu rõ ràng. |
| 6 | KIDO lên kế hoạch IPO Tường An, chia cổ tức 10%, mua lại 14,5 triệu cổ phiếu | KDC | **POSITIVE** | Chia cổ tức tiền mặt 10% và mua lại cổ phiếu — hai tín hiệu tích cực trực tiếp với cổ đông. |
| 7 | Ngành bất động sản lãi gần 80.000 tỷ đồng trong 6T2026, một tập đoàn "gánh" 80% | VHM | **POSITIVE** | VHM là tập đoàn chiếm 80% lợi nhuận toàn ngành — kết quả kinh doanh xuất sắc so với peers. |
| 8 | Tổng Giám đốc Bách Hoá Xanh đã bán ra gần 95.000 cổ phiếu Thế giới Di động | MWG | **NEGATIVE** | Lãnh đạo nội bộ bán ra cổ phiếu MWG — tín hiệu giảm niềm tin của người trong cuộc. |
| 9 | Đại gia BĐS Hải Phòng chốt ngày chào bán 200 triệu cổ phiếu giá 10.000 đồng — bằng một nửa giá trên sàn | TCH | **NEGATIVE** | Phát hành thêm giá 10.000đ trong khi thị giá ~24.000đ — pha loãng nghiêm trọng ~50%. |
| 10 | Đất Xanh muốn huy động 1.700 tỷ đồng để 'rót' cho dự án DatXanhHomes Parkview | DXG | **NEGATIVE** | Chào bán cổ phiếu riêng lẻ 93,5 triệu cp — pha loãng vốn, tín hiệu bất lợi cho cổ đông hiện hữu. |
| 11 | Mua ngoại tệ nhanh chóng trên MyVIB: Chủ động hành trình quốc tế, tránh cảnh chờ đợi, không lo tỷ giá | VIB | **NEUTRAL** | Thông báo tính năng ứng dụng nhỏ, mang tính PR sản phẩm, chưa rõ tác động doanh thu. |
| 12 | ACB – Ngân hàng duy nhất vào top 50 doanh nghiệp tiêu biểu Thành phố Hồ Chí Minh | ACB | **NEUTRAL** | Giải thưởng danh hiệu thuần, không phản ánh kết quả tài chính hay dòng tiền cụ thể. |
| 13 | Cổ đông PNJ chốt kế hoạch phát hành hơn 170 triệu cổ phiếu thưởng | PNJ | **NEUTRAL** | Cổ phiếu thưởng (không phải tiền mặt) — không tạo giá trị mới, chỉ điều chỉnh mệnh giá. |
| 14 | Từ 1-7: Doanh nghiệp chuyển khoản từ 10 triệu đồng phải xác thực khuôn mặt người đại diện pháp luật | TPB | **NEUTRAL** | Quy định hành chính NHNN áp dụng toàn ngành, không ảnh hưởng đặc thù một ngân hàng nào. |
| 15 | Lãi suất cho vay mua nhà đầu tháng 3 biến động thế nào? | ACB | **NEUTRAL** | Bài thống kê thị trường nhiều ngân hàng, không có tín hiệu rõ ràng về mã ACB cụ thể. |

---

## 5. Trường hợp khó

Phần này liệt kê các tình huống thường gây nhầm lẫn và quy tắc xử lý cứng cho từng case.

---

### Case 1: Cắt giảm nhân sự / Tái cấu trúc

**Quy tắc:** Thường gán **POSITIVE** nếu bài đóng khung theo hướng "tăng hiệu quả", "tối ưu vận hành".
Gán **NEGATIVE** nếu bài nhấn mạnh khủng hoảng, sa thải do kinh doanh sụt giảm.

| Ví dụ | Nhãn |
|---|:---:|
| "VNM cắt giảm 20% nhân sự, tái cơ cấu để tập trung vào mảng lõi" | POSITIVE |
| "VNM sa thải hàng loạt do doanh thu giảm 30% 3 quý liên tiếp" | NEGATIVE |
| "Thế Giới Di Động tăng trưởng 15% dù đóng hơn 150 cửa hàng" | POSITIVE |

> **Ghi nhớ:** Thu gọn để *tăng hiệu quả* khác với thu gọn vì *không còn lựa chọn*.

---

### Case 2: Lãnh đạo mua vào / bán ra cổ phiếu

**Quy tắc cứng:**
- Lãnh đạo nội bộ (CEO, CFO, thành viên HĐQT) **mua vào** → **POSITIVE**
- Lãnh đạo nội bộ **bán ra** → **NEGATIVE**
- Quỹ ngoại, tổ chức bên ngoài: xét theo ngữ cảnh (mua lớn thường POSITIVE, xả ròng lớn NEGATIVE).

| Ví dụ | Nhãn |
|---|:---:|
| "Chủ tịch HĐQT HDBank đăng ký mua thêm 2 triệu cổ phiếu HDB" | POSITIVE |
| "Tổng Giám đốc Bách Hoá Xanh bán ra gần 95.000 cổ phiếu MWG" | NEGATIVE |
| "Khối ngoại mua ròng 107 triệu USD, gom mạnh cổ phiếu ngân hàng" | POSITIVE |
| "FPT bị khối ngoại xả ròng 1.139 tỷ đồng trong một phiên" | NEGATIVE |

---

### Case 3: Phát hành thêm cổ phiếu (ESOP, riêng lẻ, cổ phiếu thưởng)

**Quy tắc cứng:**

| Loại phát hành | Nhãn | Điều kiện |
|---|:---:|---|
| Phát hành riêng lẻ / chào bán công chúng | NEGATIVE | Giá phát hành **thấp hơn** thị giá |
| Phát hành riêng lẻ / chào bán công chúng | POSITIVE | Giá phát hành **cao hơn** thị giá đáng kể (> 10%) |
| ESOP (cổ phiếu cho nhân viên) | NEGATIVE | Gây pha loãng, giá thường thấp hơn thị giá |
| Cổ phiếu thưởng (stock dividend) | NEUTRAL | Chỉ tách mệnh giá, không tạo giá trị mới |

> **Ví dụ thực tế:** "TCH chào bán 200 triệu cp giá 10.000đ, thị giá 24.150đ"
> → Giá phát hành chỉ bằng 41% thị giá → **NEGATIVE** rõ ràng.

---

### Case 4: Chia cổ tức

**Quy tắc cứng:**

| Loại cổ tức | Nhãn | Lý do |
|---|:---:|---|
| Cổ tức **tiền mặt** | POSITIVE | Dòng tiền thực sự chảy về tay cổ đông |
| Cổ tức **cổ phiếu** (cổ phiếu thưởng) | NEUTRAL | Không tạo giá trị mới, chỉ điều chỉnh cơ cấu vốn |
| Huỷ chia cổ tức đã cam kết | NEGATIVE | Phá vỡ kỳ vọng, tín hiệu tiêu cực về dòng tiền |

> **Ví dụ thực tế:** "VIB chia cổ tức tiền mặt 21%" → **POSITIVE**
> "PNJ phát hành 170 triệu cổ phiếu thưởng tỷ lệ 2:1" → **NEUTRAL**

---

### Case 5: Bị cảnh báo / kiểm soát / cắt margin

**Quy tắc:** Luôn gán **NEGATIVE**. Không có ngoại lệ.

| Ví dụ | Nhãn |
|---|:---:|
| "SSI đưa cổ phiếu XYZ vào diện không cho vay ký quỹ (cắt margin)" | NEGATIVE |
| "HNX đưa cổ phiếu ABC vào diện kiểm soát" | NEGATIVE |
| "UBCK yêu cầu công ty niêm yết giải trình giao dịch bất thường" | NEGATIVE |
| "Cổ phiếu bị cảnh báo do lợi nhuận âm 3 năm liên tiếp" | NEGATIVE |

---

### Case 6: Lãnh đạo bị khởi tố, bắt giữ, điều tra hình sự

**Quy tắc:** Luôn gán **NEGATIVE**, kể cả khi lãnh đạo đó đã từ nhiệm hoặc bài nhắc sự kiện quá khứ.

| Ví dụ | Nhãn |
|---|:---:|
| "Khởi tố nguyên Chủ tịch HĐQT ngân hàng vì tội vi phạm quy định cho vay" | NEGATIVE |
| "Bắt giữ Tổng Giám đốc công ty niêm yết vì tội gian lận tài chính" | NEGATIVE |
| "Điều tra CEO Mua chung BĐS Việt Nam chiếm đoạt tiền nhà đầu tư" | NEGATIVE |

> **Ngoại lệ hiếm:** Nếu bài nhắc vụ việc của công ty KHÁC, còn primary_ticker chỉ là bên liên quan gián tiếp
> → xem xét gán NEUTRAL, ghi chú giải thích.

---

### Case 7: Tin PR thuần (giải thưởng, tài trợ, ra mắt sản phẩm nhỏ)

**Quy tắc:** Gán **NEUTRAL** trừ khi có con số tài chính cụ thể và đáng kể đi kèm.

| Ví dụ | Nhãn | Lý do |
|---|:---:|---|
| "ACB được vinh danh Top 50 doanh nghiệp tiêu biểu TP.HCM" | NEUTRAL | Danh hiệu, không có giá trị tài chính cụ thể |
| "VIB ra mắt tính năng mua ngoại tệ trên app MyVIB" | NEUTRAL | Tính năng nhỏ, chưa rõ tác động doanh thu |
| "Vietnam Airlines tăng cường kết nối quốc tế, nâng tầm vị thế" | NEUTRAL | Câu PR chung, không có số liệu cụ thể |
| "Techcombank chính thức thành lập công ty bảo hiểm nhân thọ, vốn 1.300 tỷ đồng" | POSITIVE | Có con số vốn cụ thể, mở mảng kinh doanh mới rõ ràng |

> **Nguyên tắc phân biệt:** Nếu bạn hỏi "con số cụ thể là bao nhiêu?" mà bài không trả lời được → NEUTRAL.

---

### Case 8: Kết quả kinh doanh

**Quy tắc cứng:** So sánh với **cùng kỳ năm trước (YoY)**, không so với kỳ liền trước (QoQ).

| Biến động YoY | Nhãn |
|---|:---:|
| Tăng trưởng > +5% | POSITIVE |
| Đi ngang (−5% đến +5%) | NEUTRAL |
| Sụt giảm < −5% | NEGATIVE |
| Lỗ tiếp tục, không có tín hiệu phục hồi | NEGATIVE |
| Lỗ thu hẹp đáng kể (> 50%) / gần breakeven | POSITIVE |

> **Ví dụ thực tế:**
> "LPBank lợi nhuận tăng 17% năm 2025" → YoY +17% → **POSITIVE**
> "MWG doanh thu tăng 14% trong 8T2025 dù đóng 150 cửa hàng" → **POSITIVE**
> "Ngành BĐS lãi 80.000 tỷ, VHM gánh 80%" → VHM tăng mạnh → **POSITIVE**

---

### Case 9: Tin vĩ mô không gắn mã cụ thể

**Quy tắc:** Gán **NEUTRAL (0)** và **không đưa vào tập train** mô hình cổ phiếu (trường `is_macro = True`).
Các mẫu này có thể dùng huấn luyện mô hình vĩ mô riêng nếu cần.

| Ví dụ | Nhãn |
|---|:---:|
| "Đô la Mỹ tiếp tục trượt giá" (không có mã) | NEUTRAL |
| "Lãi suất liên ngân hàng VND giảm dưới 4%/năm" (không gắn ngân hàng cụ thể) | NEUTRAL |
| "Diễn biến mới nhất giá vàng SJC ngày 28/7" | NEUTRAL |
| "Bộ Tài chính nói gì về đề xuất ghi nợ tiền sử dụng đất?" | NEUTRAL |
| "Hàng triệu người Khánh Hòa đón tin vui về cao tốc" (TFC là ticker nhưng tin về hạ tầng chung) | NEUTRAL |

---

## 6. Quy tắc quyết khi phân vân

> **Nguyên tắc vàng:** Khi phân vân giữa NEUTRAL và một cực (POSITIVE hoặc NEGATIVE), **chọn NEUTRAL**.
> Chỉ gán cực khi tác động lên giá trị cổ phiếu là **rõ ràng và trực tiếp**.

### Sơ đồ quyết định nhanh

```
Đọc title + sapo
      |
      v
Có mã cổ phiếu bị ảnh hưởng trực tiếp không?
      |                       |
    Không                    Có
      |                       |
      v                       v
  NEUTRAL          Tác động lên kỳ vọng giá trị?
                     |         |           |
                   Tăng      Giảm      Không rõ
                     |         |           |
                 POSITIVE  NEGATIVE    NEUTRAL
```

### Checklist trước khi gán nhãn

- [ ] Tôi đã xác định đúng `primary_ticker` chưa?
- [ ] Tôi đang nhìn từ góc độ nhà đầu tư đang *nắm giữ* mã đó, không phải đọc theo nghĩa đen bài báo?
- [ ] Tác động lên kỳ vọng giá trị có **rõ ràng và trực tiếp** không?
- [ ] Nếu phân vân, tôi đã chọn NEUTRAL chưa?

---

## 7. Quy tắc xử lý bài nhắc nhiều mã

Nhiều bài báo nhắc đến đồng thời 2–5 mã cổ phiếu với hướng tác động khác nhau.

**Quy tắc:** Luôn gán nhãn theo `primary_ticker` — mã được xác định theo thứ tự ưu tiên:

1. `tickers_widget` — mã hiển thị trực tiếp trên widget cổ phiếu của trang báo (độ tin cậy cao nhất)
2. `tickers_explicit` — mã được nhắc đích danh trong bài ("cổ phiếu FPT", "mã VIC"...)
3. `tickers_by_name` — mã được suy ra từ tên công ty đầy đủ

**Ví dụ minh hoạ:**

> **Bài:** *"Khối ngoại bán ròng đột biến hơn 1.200 tỷ đồng, FPT bị xả mạnh nhất với 1.139 tỷ đồng"*
> `primary_ticker = FPT` → Nhìn từ góc cổ đông FPT → bị xả ròng nặng → **NEGATIVE**

> **Bài:** *"Lãi suất cho vay mua nhà tháng 3: ACB, VCB, TCB, SHB đều điều chỉnh"*
> `primary_ticker = ACB` → Tin thị trường chung, không rõ ACB được lợi hay thiệt → **NEUTRAL**

> **Bài:** *"Ngành BĐS lãi 80.000 tỷ trong 6T2026, VHM gánh 80%"*
> `primary_ticker = VHM` (chiếm 80% lợi nhuận) → tác động trực tiếp và rõ → **POSITIVE**

**Trường hợp đặc biệt:** Nếu primary_ticker không phải nhân vật chính của bài (pipeline gán sai),
người gán nhãn được phép:

1. Gán nhãn theo nhân vật chính thực tế của bài
2. Ghi chú "primary_ticker nghi sai" vào ô ghi chú
3. Báo cáo ngay về nhóm data để kiểm tra pipeline

---

*Tài liệu này được cập nhật định kỳ khi có case mới hoặc phản hồi từ vòng gán nhãn.
Mọi thắc mắc hoặc trường hợp chưa có trong hướng dẫn, vui lòng báo về nhóm dữ liệu.*
