from __future__ import annotations

import re
import unicodedata

POSITIVE_TERMS: dict[str, float] = {
    "lãi kỷ lục": 3.0, "lợi nhuận kỷ lục": 3.0, "lãi lớn": 2.0, "báo lãi": 1.0,
    "cao nhất lịch sử": 2.5, "cao nhất từ trước đến nay": 2.5, "kỷ lục mới": 2.0,
    "vượt kế hoạch": 2.2, "vượt chỉ tiêu": 2.2, "hoàn thành vượt": 2.0,
    "tăng trưởng": 1.2, "bứt phá": 1.6, "bứt tốc": 1.6, "khởi sắc": 1.4,
    "tăng gấp": 2.2, "gấp đôi": 1.8, "gấp 3": 2.0, "gấp 5": 2.2,
    "lãi tăng": 2.0, "lợi nhuận tăng": 2.0, "doanh thu tăng": 1.5,
    "chuyển từ lỗ sang lãi": 2.5, "có lãi trở lại": 2.2, "xóa lỗ lũy kế": 2.5,
    "cổ tức bằng tiền": 2.5, "cổ tức tiền mặt": 2.5, "chi trả cổ tức": 1.8,
    "tạm ứng cổ tức": 2.0, "trả cổ tức": 1.5, "chốt quyền": 0.6,
    "mua cổ phiếu quỹ": 2.0, "mua lại cổ phiếu": 1.8,
    "đăng ký mua": 1.6, "mua vào": 1.4, "gom cổ phiếu": 1.8,
    "trở thành cổ đông lớn": 1.6, "nâng sở hữu": 1.4, "tăng tỷ lệ sở hữu": 1.4,
    "tăng trần": 2.2, "tăng kịch trần": 2.2, "tím lịm": 2.0, "vượt đỉnh": 2.0,
    "lập đỉnh": 1.8, "bứt phá mạnh": 1.8,
    "ký hợp đồng": 1.6, "trúng thầu": 2.2, "trúng gói thầu": 2.2,
    "hợp đồng lớn": 1.8, "mở rộng thị trường": 1.4, "thâm nhập thị trường": 1.4,
    "được giao đất": 1.6, "chấp thuận chủ trương đầu tư": 1.4,
    "khởi công": 1.0, "hưởng lợi": 1.8, "đón tin vui": 1.8, "tin vui": 1.5,
    "nâng hạng tín nhiệm": 2.2, "xếp hạng tín nhiệm": 1.0, "triển vọng ổn định": 1.2,
    "huy động thành công": 1.4, "hoàn tất huy động": 1.4,
    "niêm yết trên hose": 1.2, "chính thức niêm yết": 1.2, "lên sàn": 0.8,
    "vào rổ": 1.6, "lọt vào vn30": 2.0, "gia nhập vn30": 2.0,
    "thuế suất 0%": 2.0, "không bị áp thuế": 2.2, "được xóa nợ": 2.2,
    "được miễn": 1.6, "giãn nợ": 1.2, "được cấp phép": 1.2,
}

NEGATIVE_TERMS: dict[str, float] = {
    "báo lỗ": -3.0, "thua lỗ": -3.0, "lỗ lũy kế": -2.5, "lỗ nặng": -3.0,
    "lỗ kỷ lục": -3.0, "chuyển từ lãi sang lỗ": -3.0,
    "lợi nhuận giảm": -2.2, "lãi giảm": -2.2, "lãi ròng giảm": -2.2,
    "giảm sốc": -2.6, "giảm sâu": -2.2, "sụt giảm": -1.8, "giảm mạnh": -1.8,
    "bốc hơi": -2.0, "đi lùi": -1.8, "hụt hơi": -1.6, "giảm tốc": -1.6,
    "không đạt kế hoạch": -1.6, "chưa như mong muốn": -1.4,
    "bị khởi tố": -3.2, "khởi tố": -3.0, "bị bắt": -3.0, "tạm giam": -3.0,
    "bị điều tra": -2.6, "vụ án": -2.4, "buôn lậu": -2.6,
    "bị xử phạt": -2.4, "bị phạt": -2.4, "xử phạt vi phạm": -2.4,
    "truy thu thuế": -2.0, "vi phạm công bố thông tin": -2.0,
    "tuýt còi": -2.2, "bị cảnh báo": -2.2, "diện cảnh báo": -2.2,
    "diện kiểm soát": -2.4, "đình chỉ giao dịch": -3.0, "tạm ngừng giao dịch": -2.8,
    "hủy niêm yết": -3.0, "hủy tư cách đại chúng": -2.2,
    "cắt margin": -2.4, "hạ tỷ lệ margin": -2.2, "không đủ điều kiện giao dịch ký quỹ": -2.4,
    "ý kiến ngoại trừ": -2.2, "sai sót trọng yếu": -2.4, "cưỡng chế": -2.4,
    "phong tỏa tài sản": -2.8, "tạm hoãn xuất cảnh": -2.6, "nợ thuế": -2.0,
    "chậm đóng bảo hiểm": -2.0, "nợ bảo hiểm": -2.0,
    "đăng ký bán": -1.8, "bán ra": -1.6, "thoái vốn": -1.2, "xả hàng": -2.2,
    "không còn là cổ đông lớn": -1.8, "rời ghế cổ đông lớn": -1.8,
    "giảm sở hữu": -1.4, "bán giải chấp": -2.6, "call margin": -2.6,
    "bán tháo": -2.6, "bán ròng": -1.6,
    "giảm sàn": -2.6, "nằm sàn": -2.6, "lao dốc": -2.4, "thủng đáy": -2.6,
    "giảm hết biên độ": -2.6, "giảm kịch biên độ": -2.6, "mất vốn hóa": -2.2,
    "pha loãng": -2.0, "chậm thanh toán": -2.6, "khất nợ": -2.6,
    "chậm trả gốc": -2.6, "chậm trả lãi": -2.6, "mất khả năng thanh toán": -3.0,
    "lùi tiến độ": -1.8, "chậm tiến độ": -1.8, "dừng nhà máy": -2.4,
    "đóng cửa": -1.8, "giải thể": -1.8, "phá sản": -3.0,
    "trích lập dự phòng": -1.6, "nợ xấu": -1.6, "kiện tụng": -1.8,
}

NEUTRAL_TERMS: dict[str, float] = {
    "vinh danh": 1.0, "giải thưởng": 1.0, "được trao": 0.8, "bằng khen": 1.2,
    "top 10": 0.8, "top 50": 0.8, "danh hiệu": 1.0, "kỷ niệm": 1.0,
    "ra mắt tính năng": 1.2, "nâng cấp ứng dụng": 1.2, "ưu đãi": 1.0,
    "khuyến mãi": 1.2, "tri ân khách hàng": 1.4, "trải nghiệm": 0.8,
    "bổ nhiệm": 1.4, "miễn nhiệm": 1.0, "từ nhiệm": 1.0, "thay đổi nhân sự": 1.4,
    "kiện toàn nhân sự": 1.4, "đhđcđ thường niên": 1.0, "đại hội đồng cổ đông": 0.8,
    "công bố lãi suất": 1.6, "lãi suất huy động": 1.6, "biểu lãi suất": 1.6,
    "cổ phiếu thưởng": 1.6, "cổ tức bằng cổ phiếu": 1.6, "trả cổ tức bằng cổ phiếu": 1.6,
    "esop": 1.0, "thay đổi địa chỉ": 1.4, "chuyển trụ sở": 1.4,
    "hội thảo": 1.2, "diễn đàn": 1.0, "tài trợ": 1.0, "an sinh xã hội": 1.4,
    "cảnh báo lừa đảo": 1.6, "mạo danh": 1.4,
    "khuyến nghị": 1.2, "dự báo thị trường": 1.2, "nhận định": 1.0,
}

NEGATORS = ("không ", "chưa ", "phủ nhận ", "bác bỏ ", "không còn ")

BROKER_REPORT_RE = re.compile(
    r"\b(mbs|ssi|vndirect|vcbs|shs|bsc|vcsc|vietcap|mas|tps|vds|sgi|kbsv|agriseco)"
    r"\s+(dự báo|nhận định|đánh giá|khuyến nghị|gọi tên|chỉ ra|chỉ tên|ước tính|research)",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", str(s)).lower()


def score_text(text: str) -> dict[str, float]:
    t = _norm(text)
    pos = neg = neu = 0.0
    n_hit = 0

    for table in (POSITIVE_TERMS, NEGATIVE_TERMS):
        for term, w in table.items():
            idx = t.find(term)
            if idx < 0:
                continue
            n_hit += 1
            head = t[max(0, idx - 12): idx]
            if any(head.endswith(ng) for ng in NEGATORS):
                w = -w
            if w > 0:
                pos += w
            else:
                neg += w

    for term, w in NEUTRAL_TERMS.items():
        if term in t:
            neu += w
            n_hit += 1

    return {
        "lex_pos": pos,
        "lex_neg": neg,
        "lex_neu": neu,
        "lex_net": pos + neg,
        "lex_nhit": float(n_hit),
        "lex_report": 1.0 if BROKER_REPORT_RE.search(t) else 0.0,
    }


FEATURE_NAMES = ["lex_pos", "lex_neg", "lex_neu", "lex_net", "lex_nhit", "lex_report"]


def rule_label(text: str, margin: float = 2.0) -> str | None:
    f = score_text(text)
    if f["lex_report"] > 0:
        return "NEUTRAL"
    net = f["lex_net"]
    if net >= margin and f["lex_neu"] < 2.0:
        return "POSITIVE"
    if net <= -margin:
        return "NEGATIVE"
    if f["lex_nhit"] == 0 or (abs(net) < 1.0 and f["lex_neu"] >= 1.0):
        return "NEUTRAL"
    return None
