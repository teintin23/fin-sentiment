"""
label_tool.py — Web app gán nhãn sentiment tin tài chính tiếng Việt
Chạy: python src/label_tool.py
Cổng: 5000
"""

import os
import sys
import threading
import webbrowser
from pathlib import Path

import pandas as pd
from flask import Flask, redirect, render_template_string, request, url_for

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
CSV_PATH  = BASE_DIR / "data" / "interim" / "gold_seed_v2.csv"

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Session: chỉ dùng 1 biến toàn cục đơn giản (single-user local tool) ───────
_session = {}          # {cursor: int}  — index trong danh sách unlabeled

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_df():
    return pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype=str)


def save_df(df: pd.DataFrame):
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


def unlabeled_indices(df: pd.DataFrame):
    """Trả về list index (gốc trong df) của các dòng chưa có nhãn."""
    mask = df["label"].isna() | (df["label"].str.strip() == "")
    return df[mask].index.tolist()


def get_cursor():
    return _session.get("cursor", 0)


def set_cursor(v: int):
    _session["cursor"] = v

# ── Template HTML ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gán nhãn Sentiment</title>
<style>
  :root {
    --pos: #16a34a; --neg: #dc2626; --neu: #6b7280;
    --bg: #f8fafc; --card: #ffffff; --border: #e2e8f0;
    --text: #1e293b; --muted: #64748b;
    --ticker-bg: #1e40af; --ticker-fg: #ffffff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
         color: var(--text); display: flex; min-height: 100vh; }

  /* ── Layout ── */
  #main  { flex: 1; padding: 28px 32px; max-width: 820px; }
  #guide { width: 320px; background: var(--card); border-left: 1px solid var(--border);
           padding: 24px 20px; position: sticky; top: 0; height: 100vh; overflow-y: auto;
           font-size: 13.5px; line-height: 1.6; }

  /* ── Progress ── */
  .progress-wrap { margin-bottom: 22px; }
  .progress-label { font-size: 14px; color: var(--muted); margin-bottom: 6px; }
  .progress-bar-bg { background: var(--border); border-radius: 8px; height: 8px; }
  .progress-bar-fill { background: var(--ticker-bg); border-radius: 8px; height: 8px;
                        transition: width .4s ease; }

  /* ── Article card ── */
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
          padding: 28px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }

  .ticker-badge { display: inline-block; background: var(--ticker-bg); color: var(--ticker-fg);
                  font-size: 24px; font-weight: 800; letter-spacing: 2px;
                  padding: 6px 20px; border-radius: 8px; margin-bottom: 16px; }

  .article-title { font-size: 20px; font-weight: 700; line-height: 1.4;
                   margin-bottom: 14px; color: var(--text); }

  .article-sapo  { font-size: 15px; color: #334155; line-height: 1.7; margin-bottom: 16px; }

  .article-link  { display: inline-block; font-size: 13px; color: #3b82f6;
                   text-decoration: none; margin-bottom: 4px; }
  .article-link:hover { text-decoration: underline; }

  /* ── Buttons ── */
  .btn-row { display: flex; gap: 12px; margin: 20px 0 12px; flex-wrap: wrap; }
  .btn-label { flex: 1; min-width: 120px; padding: 16px 8px; border: none;
               border-radius: 10px; font-size: 17px; font-weight: 700;
               cursor: pointer; transition: filter .15s, transform .1s; }
  .btn-label:hover  { filter: brightness(1.08); transform: translateY(-1px); }
  .btn-label:active { transform: translateY(0); filter: brightness(.95); }
  .btn-neg  { background: var(--neg); color: #fff; }
  .btn-neu  { background: var(--neu); color: #fff; }
  .btn-pos  { background: var(--pos); color: #fff; }

  .shortcut-hint { font-size: 12px; opacity: .7; display: block; margin-top: 4px; }

  /* ── Note & back ── */
  .note-row { display: flex; gap: 10px; align-items: center; margin-top: 4px; }
  .note-input { flex: 1; padding: 9px 12px; border: 1px solid var(--border);
                border-radius: 8px; font-size: 14px; background: var(--bg); }
  .btn-back { padding: 9px 18px; background: var(--bg); border: 1px solid var(--border);
              border-radius: 8px; font-size: 14px; cursor: pointer; color: var(--muted); }
  .btn-back:hover { background: var(--border); }

  /* ── Guide panel ── */
  #guide h2 { font-size: 15px; font-weight: 700; margin-bottom: 14px; color: var(--text); }
  #guide .q  { background: #eff6ff; border-left: 3px solid #3b82f6;
                padding: 10px 12px; border-radius: 6px; margin-bottom: 16px;
                font-size: 13px; font-style: italic; color: #1e40af; }
  #guide .def-block { margin-bottom: 14px; }
  #guide .def-title { font-weight: 700; font-size: 13px; margin-bottom: 4px; }
  #guide .def-pos { color: var(--pos); }
  #guide .def-neg { color: var(--neg); }
  #guide .def-neu { color: var(--neu); }
  #guide .def-body { font-size: 12.5px; color: #475569; padding-left: 10px; }
  #guide .note-warn { margin-top: 16px; font-size: 12px; color: var(--muted);
                       border-top: 1px solid var(--border); padding-top: 12px; }

  /* ── Done screen ── */
  .done-card { max-width: 520px; margin: 80px auto; text-align: center;
               background: var(--card); border: 1px solid var(--border);
               border-radius: 16px; padding: 48px 40px;
               box-shadow: 0 4px 16px rgba(0,0,0,.08); }
  .done-emoji { font-size: 56px; margin-bottom: 16px; }
  .done-card h1 { font-size: 26px; margin-bottom: 10px; }
  .done-card p  { color: var(--muted); margin-bottom: 24px; }
  .dist-table { width: 100%; border-collapse: collapse; text-align: left; margin-top: 20px; }
  .dist-table th, .dist-table td { padding: 10px 14px; border-bottom: 1px solid var(--border); }
  .dist-table th { font-weight: 600; background: var(--bg); }
  .badge-pos { color: var(--pos); font-weight: 700; }
  .badge-neg { color: var(--neg); font-weight: 700; }
  .badge-neu { color: var(--neu); font-weight: 700; }
</style>
</head>
<body>

{% if done %}
<!-- ── DONE SCREEN ── -->
<div style="flex:1;display:flex;align-items:center;justify-content:center;">
<div class="done-card">
  <div class="done-emoji">🎉</div>
  <h1>Xong! Đã gán {{ n_total }} bài</h1>
  <p>Tất cả bài trong gold_seed_v2.csv đã được gán nhãn.</p>
  <table class="dist-table">
    <thead><tr><th>Nhãn</th><th>Số bài</th><th>Tỷ lệ</th></tr></thead>
    <tbody>
      <tr><td class="badge-pos">POSITIVE</td><td>{{ dist.POSITIVE }}</td><td>{{ "%.1f"|format(dist.POSITIVE/n_total*100) }}%</td></tr>
      <tr><td class="badge-neg">NEGATIVE</td><td>{{ dist.NEGATIVE }}</td><td>{{ "%.1f"|format(dist.NEGATIVE/n_total*100) }}%</td></tr>
      <tr><td class="badge-neu">NEUTRAL</td><td>{{ dist.NEUTRAL }}</td><td>{{ "%.1f"|format(dist.NEUTRAL/n_total*100) }}%</td></tr>
    </tbody>
  </table>
</div>
</div>

{% else %}
<!-- ── LABEL SCREEN ── -->
<div id="main">

  <!-- Progress -->
  <div class="progress-wrap">
    <div class="progress-label">Đã gán <strong>{{ labeled }}</strong> / <strong>{{ total }}</strong> bài cần gán nhãn</div>
    <div class="progress-bar-bg">
      <div class="progress-bar-fill" style="width:{{ pct }}%"></div>
    </div>
  </div>

  <!-- Article card -->
  <div class="card">
    <div class="ticker-badge">{{ row.primary_ticker }}</div>
    <div class="article-title">{{ row.title }}</div>
    <div class="article-sapo">{{ row.sapo }}</div>
    <a class="article-link" href="{{ row.url }}" target="_blank" rel="noopener">🔗 Đọc bài gốc ↗</a>
  </div>

  <!-- Label buttons -->
  <form method="POST" action="/label" id="labelForm">
    <input type="hidden" name="row_id"  value="{{ row.id }}">
    <input type="hidden" name="label"   id="hiddenLabel" value="">
    <input type="hidden" name="cursor"  value="{{ cursor }}">

    <div class="btn-row">
      <button type="button" class="btn-label btn-neg" onclick="submitLabel('NEGATIVE')">
        😟 NEGATIVE <span class="shortcut-hint">Phím 1</span>
      </button>
      <button type="button" class="btn-label btn-neu" onclick="submitLabel('NEUTRAL')">
        😐 NEUTRAL <span class="shortcut-hint">Phím 2</span>
      </button>
      <button type="button" class="btn-label btn-pos" onclick="submitLabel('POSITIVE')">
        😊 POSITIVE <span class="shortcut-hint">Phím 3</span>
      </button>
    </div>

    <div class="note-row">
      <input class="note-input" name="note" id="noteInput"
             placeholder="Ghi chú (không bắt buộc)" value="{{ row.get('note','') or '' }}" autocomplete="off">
      {% if cursor > 0 %}
      <button type="button" class="btn-back" onclick="goBack()">← Quay lại</button>
      {% endif %}
    </div>
  </form>
</div>

<!-- ── GUIDE PANEL ── -->
<div id="guide">
  <h2>📋 Quy tắc gán nhãn</h2>
  <div class="q">
    "Nếu tôi đang <strong>CẦM</strong> mã <strong>[{{ row.primary_ticker }}]</strong>,<br>
    tin này khiến tôi <strong>vui</strong> hay <strong>lo</strong>?"
  </div>

  <div class="def-block">
    <div class="def-title def-pos">😊 POSITIVE</div>
    <div class="def-body">
      Tin làm <strong>tăng kỳ vọng</strong> về giá trị / dòng tiền doanh nghiệp.<br>
      • Kết quả kinh doanh tăng &gt;5% YoY<br>
      • Ký hợp đồng lớn, mở thị trường mới<br>
      • Lãnh đạo mua vào cổ phiếu<br>
      • Cổ tức tiền mặt<br>
      • Nhận đầu tư chiến lược, nâng hạng tín nhiệm
    </div>
  </div>

  <div class="def-block">
    <div class="def-title def-neg">😟 NEGATIVE</div>
    <div class="def-body">
      Tin làm <strong>giảm kỳ vọng</strong> về giá trị / dòng tiền doanh nghiệp.<br>
      • Kết quả kinh doanh giảm &gt;5% YoY<br>
      • Lãnh đạo bán ra cổ phiếu<br>
      • Bị cảnh báo, kiểm soát, cắt margin<br>
      • Lãnh đạo bị khởi tố, điều tra<br>
      • Phát hành pha loãng giá thấp hơn thị giá
    </div>
  </div>

  <div class="def-block">
    <div class="def-title def-neu">😐 NEUTRAL</div>
    <div class="def-body">
      Thông báo hành chính, thủ tục, tin không rõ tác động, PR thuần.<br>
      • Họp ĐHCĐ thường niên<br>
      • Đổi địa chỉ, tên công ty con<br>
      • Giải thưởng danh hiệu (không có số liệu)<br>
      • Kết quả kinh doanh đi ngang (±5%)
    </div>
  </div>

  <div class="note-warn">
    ⚠️ Đánh giá tác động lên <strong>cổ phiếu</strong>, không phải cảm xúc người viết.<br><br>
    Khi phân vân NEUTRAL vs cực → chọn <strong>NEUTRAL</strong>.<br>
    Mỗi mẫu mục tiêu: 20–40 giây.
  </div>
</div>

<script>
function submitLabel(lbl) {
  document.getElementById('hiddenLabel').value = lbl;
  document.getElementById('labelForm').submit();
}
function goBack() {
  window.location.href = '/back?cursor={{ cursor }}';
}
document.addEventListener('keydown', function(e) {
  // Không kích hoạt khi đang gõ vào ô note
  if (document.activeElement === document.getElementById('noteInput')) return;
  if (e.key === '1') submitLabel('NEGATIVE');
  if (e.key === '2') submitLabel('NEUTRAL');
  if (e.key === '3') submitLabel('POSITIVE');
});
</script>
{% endif %}

</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    df  = load_df()
    ul  = unlabeled_indices(df)
    cur = get_cursor()

    total_need = len(ul) + (len(df) - len(df[df["label"].isna() | (df["label"].str.strip() == "")]))
    # Số bài đã được gán (không tính pre-existing)
    labeled_now = get_cursor()   # cursor = số bài đã gán trong session hiện tại

    # Thực tế: tổng cần gán = số dòng label rỗng ban đầu (toàn bộ ul list khi bắt đầu)
    # Ta dùng len(ul) còn lại + cursor
    total_to_label = len(ul) + cur   # tổng bài cần gán trong phiên này

    if cur >= len(ul):
        # Tất cả đã gán → done screen
        all_labeled = df[df["label"].notna() & (df["label"].str.strip() != "")]["label"]
        dist = {
            "POSITIVE": int((all_labeled == "POSITIVE").sum()),
            "NEGATIVE": int((all_labeled == "NEGATIVE").sum()),
            "NEUTRAL":  int((all_labeled == "NEUTRAL").sum()),
        }
        return render_template_string(HTML, done=True,
                                      n_total=len(all_labeled),
                                      dist=dist)

    idx  = ul[cur]
    row  = df.loc[idx].to_dict()
    pct  = round(cur / total_to_label * 100, 1) if total_to_label else 0

    return render_template_string(HTML, done=False,
                                  row=row,
                                  cursor=cur,
                                  labeled=cur,
                                  total=total_to_label,
                                  pct=pct)


@app.route("/label", methods=["POST"])
def label():
    row_id = request.form.get("row_id", "").strip()
    lbl    = request.form.get("label",  "").strip()
    note   = request.form.get("note",   "").strip()
    cursor = int(request.form.get("cursor", 0))

    if lbl not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        return redirect(url_for("index"))

    df  = load_df()
    ul  = unlabeled_indices(df)

    if cursor < len(ul):
        idx = ul[cursor]
        if str(df.loc[idx, "id"]) == row_id:
            df.loc[idx, "label"] = lbl
            df.loc[idx, "note"]  = note
            save_df(df)

    set_cursor(cursor + 1)
    return redirect(url_for("index"))


@app.route("/back")
def back():
    cursor = int(request.args.get("cursor", 1))
    new_cursor = max(0, cursor - 1)

    # Nếu bài trước đã gán, xóa nhãn để hiển thị lại
    df  = load_df()
    # Lấy danh sách unlabeled TẠI THỜI ĐIỂM TRƯỚC — ta cần rebuild
    # Cách đơn giản: lấy tất cả bài có thể gán (kể cả đã gán trong phiên này)
    # Để "quay lại", ta chỉ cần giảm cursor; bài tiếp theo sẽ là bài chưa labeled
    # Nhưng bài vừa gán cần được xóa nhãn để hiện lại
    # Ta tìm bài ở vị trí new_cursor trong danh sách unlabeled MỚI
    ul_now = unlabeled_indices(df)
    # Số bài đã gán = cursor; bài ở cursor-1 là bài vừa gán
    # Sau khi gán, ul thay đổi; bài đó đã bị remove khỏi ul
    # Để hoàn tác, ta cần biết id của bài đó
    # Vì ta không lưu history, ta tính ngược:
    # total_unlabeled_at_start = len(ul_now) + cursor (cursor bài đã gán trong phiên)
    # Bài ở index new_cursor của list ban đầu... khó tính chính xác nếu nhiều phiên
    # Giải pháp pragmatic: xóa nhãn của bài đã gán gần nhất (cursor - 1)
    # bằng cách tìm trong df bài nào có label vừa được gán
    # Ta không lưu history, dùng một file tạm để lưu stack id đã gán trong phiên
    set_cursor(new_cursor)

    # Xóa nhãn bài vừa gán để hiện lại (dùng file stack nếu có)
    stack_file = BASE_DIR / "data" / "interim" / ".label_stack.txt"
    if stack_file.exists():
        lines = stack_file.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            last_id = lines[-1]
            mask = df["id"].astype(str) == last_id
            if mask.any():
                df.loc[mask, "label"] = ""
                df.loc[mask, "note"]  = ""
                save_df(df)
            stack_file.write_text("\n".join(lines[:-1]), encoding="utf-8")

    return redirect(url_for("index"))


# Ghi đè /label để lưu stack
_orig_label = app.view_functions["label"]

@app.route("/label_with_stack", methods=["POST"])
def label_with_stack():
    """Wrapper thêm stack tracking — không dùng trực tiếp."""
    pass


# Patch label để lưu id vào stack
def _patched_label():
    row_id = request.form.get("row_id", "").strip()
    lbl    = request.form.get("label",  "").strip()
    note   = request.form.get("note",   "").strip()
    cursor = int(request.form.get("cursor", 0))

    if lbl not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        return redirect(url_for("index"))

    df  = load_df()
    ul  = unlabeled_indices(df)

    if cursor < len(ul):
        idx = ul[cursor]
        if str(df.loc[idx, "id"]) == row_id:
            df.loc[idx, "label"] = lbl
            df.loc[idx, "note"]  = note
            save_df(df)
            # Lưu vào stack
            stack_file = BASE_DIR / "data" / "interim" / ".label_stack.txt"
            with open(stack_file, "a", encoding="utf-8") as f:
                f.write(row_id + "\n")

    set_cursor(cursor + 1)
    return redirect(url_for("index"))


app.view_functions["label"] = _patched_label


# ── Main ──────────────────────────────────────────────────────────────────────

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    # Mở trình duyệt sau 1.2s
    t = threading.Timer(1.2, open_browser)
    t.daemon = True
    t.start()
    print("=" * 58)
    print("  Label Tool — VN Financial Sentiment")
    print("  http://127.0.0.1:5000")
    print("  Phím tắt: 1=NEGATIVE  2=NEUTRAL  3=POSITIVE")
    print("  Ctrl+C để tắt")
    print("=" * 58)
    app.run(host="127.0.0.1", port=5000, debug=False)
