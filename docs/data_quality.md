# Data Quality Report

**Generated:** 2026-08-19 09:25:10  
**Source:** `data/raw/articles.jsonl`  
**Output:** `data/interim/articles_clean.parquet`

---

## So dong con lai sau moi buoc

| Buoc | So dong |
|------|--------:|
| 1 – Load tong so | 23,216 |
| 2 – Sau loc title | 23,213 |
| 3 – Sau dedupe url | 23,213 |
| 4 – Sau dedupe title | 23,195 |
| 5 – Sau loc date null | 23,195 |
| 6 – Sau loc date range | 23,195 |

## Ty le macro / co ticker

| Chi so | Gia tri |
|--------|--------:|
| Ty le is_macro | 53.93% |
| Ty le co ticker | 46.07% |

## Phan bo n_tickers

| n_tickers | So bai |
|----------:|-------:|
| 0 | 12,510 |
| 1 | 7,262 |
| 2 | 2,026 |
| 3 | 683 |
| 4 | 262 |
| 5 | 166 |
| 6 | 112 |
| 7 | 96 |
| 8 | 33 |
| 9 | 22 |
| 10 | 9 |
| 11 | 3 |
| 12 | 4 |
| 13 | 4 |
| 14 | 2 |
| 15 | 1 |

## Top 20 primary_ticker

| Ticker | So bai |
|--------|-------:|
| VIC | 714 |
| BID | 564 |
| NAB | 385 |
| VHM | 375 |
| TFC | 271 |
| TDF | 267 |
| VPB | 257 |
| ACB | 254 |
| FPT | 241 |
| TCB | 240 |
| VCB | 217 |
| VND | 188 |
| NVL | 184 |
| CTG | 178 |
| HDB | 177 |
| STB | 162 |
| PNJ | 155 |
| SHB | 146 |
| VIB | 145 |
| MWG | 137 |

## Ket qua kiem tra

**PASS**

- So dong output: **23,195** (yeu cau >= 15,000)
- Ty le co ticker: **46.07%** (yeu cau >= 30%)
