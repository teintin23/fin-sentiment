# Event study — sentiment và lợi suất bất thường

Mô hình: `market_model` | Nguồn nhãn: `all` | **chỉ sự kiện không chồng lấn (±10 phiên)** | Cửa sổ ước lượng: [-130,-11] phiên, tối thiểu 60 quan sát

## 1. Mẫu sự kiện

| | |
|---|---|
| Bài gốc khớp mã có giá | 745 sự kiện (đã gộp cùng mã cùng phiên) |
| Bỏ vì chồng lấn ±10 phiên | 3,136 |
| Bỏ vì nhãn xung đột cùng phiên | 0 |
| Bỏ vì thiếu dữ liệu ước lượng | 73 |
| **Sự kiện dùng được** | **663** |
| — POSITIVE | 245 |
| — NEUTRAL | 329 |
| — NEGATIVE | 89 |

## 2. Phương pháp

Brown & Warner (1985), MacKinlay (1997). Mô hình thị trường một nhân tố ước lượng OLS trên cửa sổ [-130,-11] phiên trước sự kiện, AR = R − (α + βR_m). Tin đăng sau 15:00 hoặc ngày nghỉ tính vào phiên kế tiếp. Kiểm định t cắt ngang trên CAR.

## 3. CAAR theo nhóm nhãn

| Cửa sổ | Nhóm | n | CAAR % | t | p |
|---|---|---|---|---|---|
| [0,0] | POSITIVE | 245 | +0.479 | 2.79 | 0.0057 |
| [0,0] | NEUTRAL | 329 | +0.121 | 0.89 | 0.3716 |
| [0,0] | NEGATIVE | 89 | -0.842 | -2.67 | 0.0090 |
| [0,1] | POSITIVE | 244 | +0.841 | 3.19 | 0.0016 |
| [0,1] | NEUTRAL | 329 | +0.052 | 0.26 | 0.7921 |
| [0,1] | NEGATIVE | 89 | -0.631 | -1.43 | 0.1552 |
| [0,3] | POSITIVE | 244 | +0.461 | 1.35 | 0.1796 |
| [0,3] | NEUTRAL | 325 | -0.011 | -0.04 | 0.9671 |
| [0,3] | NEGATIVE | 89 | -0.424 | -0.75 | 0.4572 |
| [0,5] | POSITIVE | 241 | +0.098 | 0.26 | 0.7946 |
| [0,5] | NEUTRAL | 318 | +0.029 | 0.09 | 0.9293 |
| [0,5] | NEGATIVE | 88 | -1.042 | -1.60 | 0.1142 |
| [-1,1] | POSITIVE | 244 | +1.166 | 3.35 | 0.0009 |
| [-1,1] | NEUTRAL | 327 | +0.191 | 0.84 | 0.4024 |
| [-1,1] | NEGATIVE | 88 | -1.222 | -2.52 | 0.0135 |

## 4. Chênh lệch POSITIVE − NEGATIVE

Đây là bảng chính: nếu nhãn có giá trị thông tin thì chênh lệch phải dương ở các cửa sổ chứa t=0.

| Cửa sổ | POS−NEG % | t (Welch) | p | n POS | n NEG |
|---|---|---|---|---|---|
| [0,0] | +1.321 | 3.68 | 0.0003 | 245 | 89 |
| [0,1] | +1.471 | 2.87 | 0.0047 | 244 | 89 |
| [0,3] | +0.885 | 1.33 | 0.1842 | 244 | 89 |
| [0,5] | +1.139 | 1.51 | 0.1322 | 241 | 88 |
| [-1,1] | +2.388 | 4.00 | 9.0e-05 | 244 | 88 |

## 5. CAAR theo ngày

![CAAR](reports/figures/event_study_caar_no_overlap.png)

## 6. Kiểm tra giả dược — cửa sổ [-5,-1]

Cửa sổ nằm hoàn toàn trước ngày tin, lẽ ra phải bằng 0. Khác 0 có ý nghĩa nghĩa là kết quả bị nhiễm (sự kiện chồng lấn hoặc thị trường phản ứng trước).

| Nhóm | n | CAAR % | t | p |
|---|---|---|---|---|
| POSITIVE | 243 | +0.864 | 2.30 | 0.0225 |
| NEUTRAL | 319 | +0.153 | 0.56 | 0.5735 |
| NEGATIVE | 87 | -1.747 | -1.78 | 0.0792 |
| POS−NEG | 243/87 | +2.610 | 2.48 | 0.0146 |

## 7. Chồng lấn sự kiện

Mẫu đã lọc chỉ giữ sự kiện không có sự kiện khác cùng mã trong ±10 phiên (còn lại 0% chồng lấn — theo thiết kế là 0). Đây là biến thể sạch để đối chiếu với kết quả chính.

## 8. Hạn chế và cách đọc p-value

- Vì chồng lấn ở mục 7, **t-statistic bị thổi phồng, p-value thật lớn hơn con số in ra**. Đừng đọc `p < 0.05` ở đây như một thí nghiệm sạch.
- Nếu kết quả không có ý nghĩa, bốn khả năng chưa loại trừ được:
  1. Nhãn quá nhiễu (kappa 0.55, đa số là nhãn lan truyền)
  2. Gán mã sai ~15%
  3. Thị trường phản ứng trước khi tin lên báo
  4. Cỡ mẫu chưa đủ, nhất là nhóm NEGATIVE

