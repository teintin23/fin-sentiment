# Event study — sentiment và lợi suất bất thường

Mô hình: `market_model` | Nguồn nhãn: `human` | Cửa sổ ước lượng: [-130,-11] phiên, tối thiểu 60 quan sát

## 1. Mẫu sự kiện

| | |
|---|---|
| Bài gốc khớp mã có giá | 120 sự kiện (đã gộp cùng mã cùng phiên) |
| Bỏ vì nhãn xung đột cùng phiên | 0 |
| Bỏ vì thiếu dữ liệu ước lượng | 6 |
| **Sự kiện dùng được** | **114** |
| — POSITIVE | 64 |
| — NEUTRAL | 30 |
| — NEGATIVE | 20 |

## 2. Phương pháp

Brown & Warner (1985), MacKinlay (1997). Mô hình thị trường một nhân tố ước lượng OLS trên cửa sổ [-130,-11] phiên trước sự kiện, AR = R − (α + βR_m). Tin đăng sau 15:00 hoặc ngày nghỉ tính vào phiên kế tiếp. Kiểm định t cắt ngang trên CAR.

## 3. CAAR theo nhóm nhãn

| Cửa sổ | Nhóm | n | CAAR % | t | p |
|---|---|---|---|---|---|
| [0,0] | POSITIVE | 64 | +0.748 | 1.91 | 0.0601 |
| [0,0] | NEUTRAL | 30 | +0.192 | 0.41 | 0.6846 |
| [0,0] | NEGATIVE | 20 | -1.463 | -2.11 | 0.0481 |
| [0,1] | POSITIVE | 64 | +0.813 | 1.26 | 0.2133 |
| [0,1] | NEUTRAL | 30 | +0.042 | 0.07 | 0.9482 |
| [0,1] | NEGATIVE | 20 | -1.483 | -1.31 | 0.2043 |
| [0,3] | POSITIVE | 64 | +0.263 | 0.33 | 0.7441 |
| [0,3] | NEUTRAL | 30 | -0.455 | -0.58 | 0.5674 |
| [0,3] | NEGATIVE | 20 | -3.498 | -1.65 | 0.1164 |
| [0,5] | POSITIVE | 64 | -0.392 | -0.39 | 0.6975 |
| [0,5] | NEUTRAL | 30 | +0.019 | 0.02 | 0.9851 |
| [0,5] | NEGATIVE | 20 | -3.968 | -1.92 | 0.0697 |
| [-1,1] | POSITIVE | 64 | +1.564 | 1.98 | 0.0526 |
| [-1,1] | NEUTRAL | 30 | -0.046 | -0.07 | 0.9476 |
| [-1,1] | NEGATIVE | 20 | -2.980 | -1.77 | 0.0926 |

## 4. Chênh lệch POSITIVE − NEGATIVE

Đây là bảng chính: nếu nhãn có giá trị thông tin thì chênh lệch phải dương ở các cửa sổ chứa t=0.

| Cửa sổ | POS−NEG % | t (Welch) | p | n POS | n NEG |
|---|---|---|---|---|---|
| [0,0] | +2.211 | 2.78 | 0.0090 | 64 | 20 |
| [0,1] | +2.297 | 1.77 | 0.0869 | 64 | 20 |
| [0,3] | +3.761 | 1.66 | 0.1106 | 64 | 20 |
| [0,5] | +3.576 | 1.56 | 0.1304 | 64 | 20 |
| [-1,1] | +4.543 | 2.44 | 0.0211 | 64 | 20 |

## 5. CAAR theo ngày

![CAAR](reports/figures/event_study_caar_human.png)

## 6. Kiểm tra giả dược — cửa sổ [-5,-1]

Cửa sổ nằm hoàn toàn trước ngày tin, lẽ ra phải bằng 0. Khác 0 có ý nghĩa nghĩa là kết quả bị nhiễm (sự kiện chồng lấn hoặc thị trường phản ứng trước).

| Nhóm | n | CAAR % | t | p |
|---|---|---|---|---|
| POSITIVE | 63 | +0.208 | 0.28 | 0.7836 |
| NEUTRAL | 30 | -1.281 | -1.53 | 0.1371 |
| NEGATIVE | 20 | -2.178 | -0.96 | 0.3479 |
| POS−NEG | 63/20 | +2.385 | 1.00 | 0.3274 |

## 7. Chồng lấn sự kiện

**12%** sự kiện có sự kiện khác cùng mã trong vòng ±10 phiên. AR của tin trước tràn vào cửa sổ của tin sau, các quan sát không độc lập.

## 8. Hạn chế và cách đọc p-value

- Vì chồng lấn ở mục 7, **t-statistic bị thổi phồng, p-value thật lớn hơn con số in ra**. Đừng đọc `p < 0.05` ở đây như một thí nghiệm sạch.
- Nếu kết quả không có ý nghĩa, bốn khả năng chưa loại trừ được:
  1. Nhãn quá nhiễu (kappa 0.55, đa số là nhãn lan truyền)
  2. Gán mã sai ~15%
  3. Thị trường phản ứng trước khi tin lên báo
  4. Cỡ mẫu chưa đủ, nhất là nhóm NEGATIVE

