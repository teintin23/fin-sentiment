# Cách áp dụng (5 phút, copy-paste từng khối)

## 1. Chép file vào repo trên máy bạn
Giải nén zip này, rồi copy đè toàn bộ vào thư mục repo `fin-sentiment` của bạn
(giữ nguyên cấu trúc thư mục: README.md ở gốc, mấy file .py vào src/, .md vào docs/).

## 2. Commit + push
```
cd fin-sentiment
git add README.md src/build_docs.py src/event_study_momentum.py docs/event_study_momentum.md docs/research_summary.md
git commit -m "Add momentum-controlled event study, research summary"
git push
```

## 3. Trên trang GitHub của repo (không làm được qua git)
Vào repo > nút bánh răng cạnh "About" bên phải > điền:

Description:
> Vietnamese financial news sentiment (5.4k CafeF articles, holder-perspective labels) + PhoBERT. Event study: announcement-day effect survives momentum controls; no post-drift.

Topics: `vietnamese` `nlp` `sentiment-analysis` `finance` `phobert` `event-study` `dataset`

## 4. Kiểm tra lại (không bắt buộc)
Muốn tự tái tạo số liệu momentum trên máy:
```
pip install statsmodels
python src/event_study_momentum.py    # ghi docs/event_study_momentum.md
python src/build_docs.py              # sinh lại README
```
Số phải ra y hệt vì dữ liệu đã commit trong repo.

## Khi email lab, dẫn link theo thứ tự này
1. docs/research_summary.md  (có abstract tiếng Anh)
2. docs/event_study_momentum.md
3. README (mục Quyết định thiết kế + Hạn chế đã biết)
