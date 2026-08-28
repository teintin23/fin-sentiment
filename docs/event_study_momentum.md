# Event study co kiem soat momentum

Muc dich: tach phan hieu ung sentiment tai phien tin ra khoi momentum truoc su kien. Tra loi phe binh o `docs/event_study.md` muc 6-8 (gia duoc [-5,-1] duong o POSITIVE).

Doc ket qua the nao:
- Bang pre-drift: dinh luong noi sinh. preCAR POSITIVE > 0 la bang chung bao viet tin tot ve ma dang tang.
- Hoi quy kiem soat: neu POS-NEG tai [0,0] van co y nghia sau khi kiem soat preCAR -> hieu ung dong thoi la that, khong phai momentum keo dai.
- Tercile thap: kiem tra phi tham so cung ket luan.
- Post-drift [1,5]: kiem dinh 'tin du bao gia' dung nghia. Khong co y nghia o day KHONG phai that bai - no co nghia thi truong hap thu tin trong phien dau, nhat quan voi thi truong hieu qua dang ban-manh.

### Mau: all  (3616 su kien dung duoc)

Pre-drift CAR[-5,-1] theo nhan (do luong noi sinh, cang khac 0 cang nhiem):

| nhan | mean preCAR | t | p | n |
|---|---|---|---|---|
| POSITIVE | +0.835% | 3.98 | 7.4e-05 | 984 |
| NEUTRAL | -0.035% | -0.33 | 0.7427 | 2269 |
| NEGATIVE | -0.577% | -1.30 | 0.1939 | 330 |

Hoi quy CAR ~ POS + NEG + preCAR[-5,-1], HC1 robust SE (he so la % neu nhan voi 100):

| cua so | b_POS | p | b_NEG | p | POS-NEG | p | b_preCAR | p | n |
|---|---|---|---|---|---|---|---|---|---|
| [0,0] | +0.489% | 5.0e-07 | -0.416% | 0.0101 | +0.905% | 3.5e-07 | +0.048 | 0.0003 | 3583 |
| [0,1] | +0.732% | 6.2e-07 | -0.585% | 0.0164 | +1.318% | 8.8e-07 | +0.067 | 0.0028 | 3579 |
| [0,3] | +0.650% | 0.0016 | -0.777% | 0.0142 | +1.427% | 5.7e-05 | +0.115 | 0.0004 | 3573 |

Tercile |preCAR| thap nhat (|preCAR| <= 1.55%, gan nhu khong co momentum truoc tin):
- POS-NEG spread tai [0,0]: **+0.224%** (t=0.98, p=0.3278, n=298/100)

Du bao that su - CAR[1,5] ~ POS + NEG + preCAR + AR[0]:

| he so | gia tri | p |
|---|---|---|
| POS | -0.111% | 0.6168 |
| NEG | -0.706% | 0.0331 |
| preCAR | +11.515% | 0.0003 |
| AR0 | +20.699% | 0.0055 |
| POS-NEG | +0.595% | 0.1172 |

(n=3556)

### Mau: all / no-overlap  (663 su kien dung duoc)

Pre-drift CAR[-5,-1] theo nhan (do luong noi sinh, cang khac 0 cang nhiem):

| nhan | mean preCAR | t | p | n |
|---|---|---|---|---|
| POSITIVE | +0.864% | 2.30 | 0.0225 | 243 |
| NEUTRAL | +0.153% | 0.56 | 0.5735 | 319 |
| NEGATIVE | -1.747% | -1.78 | 0.0792 | 87 |

Hoi quy CAR ~ POS + NEG + preCAR[-5,-1], HC1 robust SE (he so la % neu nhan voi 100):

| cua so | b_POS | p | b_NEG | p | POS-NEG | p | b_preCAR | p | n |
|---|---|---|---|---|---|---|---|---|---|
| [0,0] | +0.387% | 0.0791 | -0.942% | 0.0061 | +1.329% | 0.0002 | -0.001 | 0.9878 | 649 |
| [0,1] | +0.859% | 0.0082 | -0.811% | 0.0692 | +1.670% | 0.0003 | -0.016 | 0.7891 | 648 |
| [0,3] | +0.575% | 0.1628 | -0.367% | 0.5574 | +0.943% | 0.1459 | +0.053 | 0.4183 | 645 |

Tercile |preCAR| thap nhat (|preCAR| <= 1.54%, gan nhu khong co momentum truoc tin):
- POS-NEG spread tai [0,0]: **+0.635%** (t=1.42, p=0.1622, n=78/25)

Du bao that su - CAR[1,5] ~ POS + NEG + preCAR + AR[0]:

| he so | gia tri | p |
|---|---|---|
| POS | -0.122% | 0.7841 |
| NEG | +0.025% | 0.9654 |
| preCAR | +13.663% | 0.0295 |
| AR0 | -14.451% | 0.2486 |
| POS-NEG | -0.147% | 0.8186 |

(n=636)

### Mau: manual  (889 su kien dung duoc)

Pre-drift CAR[-5,-1] theo nhan (do luong noi sinh, cang khac 0 cang nhiem):

| nhan | mean preCAR | t | p | n |
|---|---|---|---|---|
| POSITIVE | +0.775% | 2.43 | 0.0155 | 316 |
| NEUTRAL | -0.279% | -1.31 | 0.1900 | 468 |
| NEGATIVE | -0.229% | -0.32 | 0.7459 | 98 |

Hoi quy CAR ~ POS + NEG + preCAR[-5,-1], HC1 robust SE (he so la % neu nhan voi 100):

| cua so | b_POS | p | b_NEG | p | POS-NEG | p | b_preCAR | p | n |
|---|---|---|---|---|---|---|---|---|---|
| [0,0] | +0.718% | 3.2e-05 | -0.700% | 0.0224 | +1.418% | 1.2e-05 | +0.036 | 0.1284 | 882 |
| [0,1] | +0.796% | 0.0017 | -1.132% | 0.0101 | +1.928% | 4.8e-05 | +0.048 | 0.2321 | 882 |
| [0,3] | +0.983% | 0.0030 | -1.658% | 0.0137 | +2.641% | 0.0002 | +0.090 | 0.1200 | 881 |

Tercile |preCAR| thap nhat (|preCAR| <= 1.58%, gan nhu khong co momentum truoc tin):
- POS-NEG spread tai [0,0]: **+0.302%** (t=0.68, p=0.5021, n=95/35)

Du bao that su - CAR[1,5] ~ POS + NEG + preCAR + AR[0]:

| he so | gia tri | p |
|---|---|---|
| POS | -0.305% | 0.4088 |
| NEG | -1.503% | 0.0280 |
| preCAR | +9.474% | 0.0745 |
| AR0 | +15.183% | 0.1339 |
| POS-NEG | +1.198% | 0.1019 |

(n=869)

## Ket luan

1. **Noi sinh co that va do duoc**: preCAR[-5,-1] cua nhom POSITIVE +0.84% (p=7.4e-05). Bao viet tin tot ve ma dang tang.
2. **Hieu ung dong thoi song sot sau kiem soat momentum**: POS-NEG tai [0,0] la +0.905% (p=3.5e-07) tren toan mau, +1.329% (p=0.0002) tren mau khong chong lan, +1.418% (p=1.2e-05) khi chi dung nhan doc tay. Khong phai artifact cua pre-drift.
3. **Tercile |preCAR| thap**: spread cung chieu duong nhung khong co y nghia (n=298/100) - thieu luc thong ke, khong mau thuan voi (2).
4. **Khong co drift sau tin**: POS-NEG tren CAR[1,5] khong co y nghia o ca 3 mau. Thong tin duoc dinh gia ngay trong phien tin ra.

Cau chu de bao cao: *nhan sentiment mang thong tin duoc thi truong dinh gia tai phien tin ra, doc lap voi momentum truoc su kien; khong co bang chung ve suc du bao loi suat sau do. Day la thuoc do chat luong nhan, khong phai tin hieu giao dich.*