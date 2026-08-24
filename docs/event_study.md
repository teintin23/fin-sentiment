# Event study — sentiment và lợi suất bất thường

Mô hình: `market_model` | Nguồn nhãn: `all` | Cửa sổ ước lượng: [-130,-11] phiên, tối thiểu 60 quan sát

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

Brown & Warner (1985), MacKinlay (1997). Mô hình thị trường một nhân tố ước lượng OLS trên cửa sổ [-130,-11] phiên trước sự kiện, AR = R − (α + βR_m). Tin đăng sau 15:00 hoặc ngày nghỉ tính vào phiên kế tiếp. Kiểm định t cắt ngang trên CAR.

## 3. CAAR theo nhóm nhãn

| Cửa sổ | Nhóm | n | CAAR % | t | p |
|---|---|---|---|---|---|
| [0,0] | POSITIVE | 992 | +0.526 | 6.05 | 2.1e-09 |
| [0,0] | NEUTRAL | 2289 | -0.002 | -0.04 | 0.9679 |
| [0,0] | NEGATIVE | 335 | -0.447 | -2.94 | 0.0035 |
| [0,1] | POSITIVE | 991 | +0.711 | 5.36 | 1.0e-07 |
| [0,1] | NEUTRAL | 2286 | -0.060 | -0.89 | 0.3725 |
| [0,1] | NEGATIVE | 334 | -0.646 | -2.80 | 0.0055 |
| [0,3] | POSITIVE | 990 | +0.620 | 3.31 | 0.0010 |
| [0,3] | NEUTRAL | 2279 | -0.126 | -1.31 | 0.1916 |
| [0,3] | NEGATIVE | 333 | -0.929 | -3.09 | 0.0022 |
| [0,5] | POSITIVE | 986 | +0.468 | 2.06 | 0.0393 |
| [0,5] | NEUTRAL | 2264 | -0.154 | -1.31 | 0.1918 |
| [0,5] | NEGATIVE | 331 | -1.416 | -3.83 | 0.0002 |
| [-1,1] | POSITIVE | 988 | +1.087 | 6.39 | 2.6e-10 |
| [-1,1] | NEUTRAL | 2283 | -0.072 | -0.87 | 0.3866 |
| [-1,1] | NEGATIVE | 331 | -0.934 | -3.07 | 0.0024 |

## 4. Chênh lệch POSITIVE − NEGATIVE

Đây là bảng chính: nếu nhãn có giá trị thông tin thì chênh lệch phải dương ở các cửa sổ chứa t=0.

| Cửa sổ | POS−NEG % | t (Welch) | p | n POS | n NEG |
|---|---|---|---|---|---|
| [0,0] | +0.972 | 5.55 | 4.3e-08 | 992 | 335 |
| [0,1] | +1.356 | 5.10 | 4.7e-07 | 991 | 334 |
| [0,3] | +1.548 | 4.37 | 1.5e-05 | 990 | 333 |
| [0,5] | +1.884 | 4.35 | 1.6e-05 | 986 | 331 |
| [-1,1] | +2.021 | 5.79 | 1.2e-08 | 988 | 331 |

## 5. CAAR theo ngày

![CAAR](reports/figures/event_study_caar.png)

## 6. Kiểm tra giả dược — cửa sổ [-5,-1]

Cửa sổ nằm hoàn toàn trước ngày tin, lẽ ra phải bằng 0. Khác 0 có ý nghĩa nghĩa là kết quả bị nhiễm (sự kiện chồng lấn hoặc thị trường phản ứng trước).

| Nhóm | n | CAAR % | t | p |
|---|---|---|---|---|
| POSITIVE | 984 | +0.835 | 3.98 | 7.4e-05 |
| NEUTRAL | 2269 | -0.035 | -0.33 | 0.7427 |
| NEGATIVE | 330 | -0.577 | -1.30 | 0.1939 |
| POS−NEG | 984/330 | +1.412 | 2.88 | 0.0042 |

## 7. Chồng lấn sự kiện

**81%** sự kiện có sự kiện khác cùng mã trong vòng ±10 phiên. AR của tin trước tràn vào cửa sổ của tin sau, các quan sát không độc lập.

## 8. Hạn chế và cách đọc p-value

- Vì chồng lấn ở mục 7, **t-statistic bị thổi phồng, p-value thật lớn hơn con số in ra**. Đừng đọc `p < 0.05` ở đây như một thí nghiệm sạch.
- Nếu kết quả không có ý nghĩa, bốn khả năng chưa loại trừ được:
  1. Nhãn quá nhiễu (kappa 0.55, đa số là nhãn lan truyền)
  2. Gán mã sai ~15%
  3. Thị trường phản ứng trước khi tin lên báo
  4. Cỡ mẫu chưa đủ, nhất là nhóm NEGATIVE

