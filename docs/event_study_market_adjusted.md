# Event study — sentiment và lợi suất bất thường

Mô hình: `market_adjusted` | Nguồn nhãn: `all` | Cửa sổ ước lượng: [-130,-11] phiên, tối thiểu 60 quan sát

## 1. Mẫu sự kiện

| | |
|---|---|
| Bài gốc khớp mã có giá | 4,099 sự kiện (đã gộp cùng mã cùng phiên) |
| Bỏ vì nhãn xung đột cùng phiên | 218 |
| Bỏ vì thiếu dữ liệu ước lượng | 234 |
| **Sự kiện dùng được** | **3,616** |
| — POSITIVE | 992 |
| — NEUTRAL | 2,289 |
| — NEGATIVE | 335 |

## 2. Phương pháp

Brown & Warner (1985), MacKinlay (1997). Market-adjusted: AR = R − R_m, không hồi quy. Tin đăng sau 15:00 hoặc ngày nghỉ tính vào phiên kế tiếp. Kiểm định t cắt ngang trên CAR.

## 3. CAAR theo nhóm nhãn

| Cửa sổ | Nhóm | n | CAAR % | t | p |
|---|---|---|---|---|---|
| [0,0] | POSITIVE | 992 | +0.544 | 6.27 | 5.3e-10 |
| [0,0] | NEUTRAL | 2289 | -0.019 | -0.41 | 0.6835 |
| [0,0] | NEGATIVE | 335 | -0.522 | -3.44 | 0.0007 |
| [0,1] | POSITIVE | 991 | +0.726 | 5.36 | 1.0e-07 |
| [0,1] | NEUTRAL | 2286 | -0.107 | -1.59 | 0.1124 |
| [0,1] | NEGATIVE | 334 | -0.813 | -3.46 | 0.0006 |
| [0,3] | POSITIVE | 990 | +0.657 | 3.43 | 0.0006 |
| [0,3] | NEUTRAL | 2279 | -0.235 | -2.43 | 0.0151 |
| [0,3] | NEGATIVE | 333 | -1.240 | -3.92 | 0.0001 |
| [0,5] | POSITIVE | 986 | +0.524 | 2.25 | 0.0249 |
| [0,5] | NEUTRAL | 2264 | -0.323 | -2.75 | 0.0060 |
| [0,5] | NEGATIVE | 331 | -1.803 | -4.81 | 2.3e-06 |
| [-1,1] | POSITIVE | 988 | +1.085 | 6.29 | 4.7e-10 |
| [-1,1] | NEUTRAL | 2283 | -0.143 | -1.73 | 0.0830 |
| [-1,1] | NEGATIVE | 331 | -1.180 | -3.92 | 0.0001 |

## 4. Chênh lệch POSITIVE − NEGATIVE

Đây là bảng chính: nếu nhãn có giá trị thông tin thì chênh lệch phải dương ở các cửa sổ chứa t=0.

| Cửa sổ | POS−NEG % | t (Welch) | p | n POS | n NEG |
|---|---|---|---|---|---|
| [0,0] | +1.066 | 6.09 | 2.1e-09 | 992 | 335 |
| [0,1] | +1.539 | 5.67 | 2.3e-08 | 991 | 334 |
| [0,3] | +1.897 | 5.13 | 4.0e-07 | 990 | 333 |
| [0,5] | +2.327 | 5.27 | 1.9e-07 | 986 | 331 |
| [-1,1] | +2.265 | 6.53 | 1.5e-10 | 988 | 331 |

## 5. CAAR theo ngày

![CAAR](reports/figures/event_study_caar_market_adjusted.png)

## 6. Kiểm tra giả dược — cửa sổ [-5,-1]

Cửa sổ nằm hoàn toàn trước ngày tin, lẽ ra phải bằng 0. Khác 0 có ý nghĩa nghĩa là kết quả bị nhiễm (sự kiện chồng lấn hoặc thị trường phản ứng trước).

| Nhóm | n | CAAR % | t | p |
|---|---|---|---|---|
| POSITIVE | 984 | +0.832 | 3.88 | 0.0001 |
| NEUTRAL | 2269 | -0.175 | -1.68 | 0.0930 |
| NEGATIVE | 330 | -0.906 | -2.13 | 0.0343 |
| POS−NEG | 984/330 | +1.737 | 3.64 | 0.0003 |

## 7. Chồng lấn sự kiện

**81%** sự kiện có sự kiện khác cùng mã trong vòng ±10 phiên. AR của tin trước tràn vào cửa sổ của tin sau, các quan sát không độc lập.

## 8. Hạn chế và cách đọc p-value

- Vì chồng lấn ở mục 7, **t-statistic bị thổi phồng, p-value thật lớn hơn con số in ra**. Đừng đọc `p < 0.05` ở đây như một thí nghiệm sạch.
- Nếu kết quả không có ý nghĩa, bốn khả năng chưa loại trừ được:
  1. Nhãn quá nhiễu (kappa 0.55, đa số là nhãn lan truyền)
  2. Gán mã sai ~15%
  3. Thị trường phản ứng trước khi tin lên báo
  4. Cỡ mẫu chưa đủ, nhất là nhóm NEGATIVE

