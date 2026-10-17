"""图像编解码与叠加渲染。

说明：OpenCV 的 Hershey 字体不支持中文，因此所有文字叠加统一走 PIL + 系统 CJK 字体；
无 PIL 或字体时才退回 cv2.putText（此时中文会显示为占位符，但流程不中断）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # noqa: BLE001
    Image = ImageDraw = ImageFont = None

# 颜色约定（BGR）：检测=蓝系、分割=绿系、测距=橙/红蓝
COLOR_DETECT = (255, 160, 60)
COLOR_SEG = (80, 220, 120)
COLOR_MIN = (60, 60, 255)      # 最近点：红
COLOR_MAX = (255, 140, 40)     # 最远点：蓝
COLOR_SAMPLE = (90, 230, 90)   # 普通采样点：绿
COLOR_LEFT = (255, 200, 0)     # 左边界：青
COLOR_RIGHT = (200, 0, 255)    # 右边界：品红
COLOR_TEXT = (255, 255, 255)
COLOR_TITLE = (0, 220, 255)
COLOR_WARN = (0, 200, 255)


@dataclass(frozen=True)
class TextStyle:
    """单行文字的绘制样式（颜色 / 字号 / 线宽）。"""

    color: tuple[int, int, int] = COLOR_TEXT
    scale: float = 0.5
    thickness: int = 1


@dataclass(frozen=True)
class HudStyle:
    """HUD 多行信息面板的样式（行高 / 颜色 / 字号）。"""

    line_height: int = 26
    color: tuple[int, int, int] = COLOR_TEXT
    size: int = 16


_FONT_CANDIDATES = (
    "msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf",
    "NotoSansCJK-Regular.ttc", "wqy-zenhei.ttc",
)
_FONT_DIRS = (Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/System/Library/Fonts"))
_font_cache: dict[int, object] = {}


def _load_font(size: int):
    if ImageFont is None:
        return None
    size = max(8, int(size))
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for folder in _FONT_DIRS:
        if not folder.exists():
            continue
        for name in _FONT_CANDIDATES:
            path = folder / name
            if path.exists():
                try:
                    font = ImageFont.truetype(str(path), size)
                    break
                except Exception:  # noqa: BLE001
                    continue
        if font is not None:
            break
    _font_cache[size] = font
    return font


def text_size(text: str, size: int = 16) -> tuple[int, int]:
    font = _load_font(size)
    if font is None:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 28.0, 1)
        return int(w), int(h)
    box = font.getbbox(text)
    return int(box[2] - box[0]), int(box[3] - box[1])


def draw_text_items(image: np.ndarray, items: list[dict]) -> np.ndarray:
    """批量绘制文字（单次 PIL 往返，避免每行一次图像转换）。

    items: [{"text": str, "xy": (x, y), "color": (b,g,r), "size": int,
             "bg": (b,g,r) | None, "padding": int}]
    """
    if not items:
        return image
    font = _load_font(items[0].get("size", 16))
    if font is None or Image is None:
        for item in items:
            cv2.putText(image, str(item["text"]), tuple(int(v) for v in item["xy"]),
                        cv2.FONT_HERSHEY_SIMPLEX, item.get("size", 16) / 28.0,
                        tuple(int(c) for c in item.get("color", COLOR_TEXT)), 1, cv2.LINE_AA)
        return image

    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(pil)
    for item in items:
        text = str(item["text"])
        x, y = int(item["xy"][0]), int(item["xy"][1])
        size = int(item.get("size", 16))
        item_font = _load_font(size) or font
        color = tuple(int(c) for c in item.get("color", COLOR_TEXT))
        rgb = (color[2], color[1], color[0])
        bg = item.get("bg")
        if bg is not None:
            pad = int(item.get("padding", 4))
            w, h = text_size(text, size)
            bg_rgb = (int(bg[2]), int(bg[1]), int(bg[0]))
            drawer.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], fill=bg_rgb)
        drawer.text((x, y), text, font=item_font, fill=rgb)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def put_label(image: np.ndarray, text: str, org: tuple[int, int],
              style: TextStyle = TextStyle()) -> None:
    """兼容旧调用：单行文字。"""
    size = int(round(max(11.0, float(style.scale) * 28.0)))
    draw_text_items(image, [{"text": text, "xy": (int(org[0]), int(org[1]) - size),
                             "color": style.color, "size": size}])


# ------------------------------------------------------------------ 基础 IO
def decode_image(buf: bytes) -> np.ndarray | None:
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def read_image(path: str | Path) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img


def read_depth16(path: str | Path, depth_scale: float = 0.001) -> np.ndarray | None:
    """读取 16 位深度 PNG，返回米为单位的 float32 数组。"""
    p = Path(path)
    if not p.exists():
        return None
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return raw.astype(np.float32) * float(depth_scale)


def write_depth16(path: str | Path, depth_m: np.ndarray, depth_scale: float = 0.001) -> Path:
    """深度图必须以 16 位 PNG 无损保存（不能存成 8 位可视化图）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = np.clip(np.asarray(depth_m, dtype=np.float32) / float(depth_scale), 0, 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", raw)
    if not ok:
        raise RuntimeError(f"深度图编码失败: {p}")
    buf.tofile(str(p))
    return p


def write_image(path: str | Path, image: np.ndarray, quality: int = 95) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"图像编码失败: {p}")
    buf.tofile(str(p))
    return p


def encode_jpeg(image: np.ndarray, quality: int = 75) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""


def colorize_depth(depth_m: np.ndarray, min_m: float = 0.3, max_m: float = 12.0) -> np.ndarray:
    """深度伪彩图：近处偏红、远处偏蓝；无效点为黑。"""
    arr = np.asarray(depth_m, dtype=np.float32)
    valid = (arr > 0) & np.isfinite(arr)
    norm = np.zeros_like(arr, dtype=np.float32)
    span = max(max_m - min_m, 1e-3)
    norm[valid] = np.clip((arr[valid] - min_m) / span, 0.0, 1.0)
    gray = ((1.0 - norm) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def resize_keep_ratio(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    if abs(scale - 1.0) < 1e-3:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ 叠加
def draw_detections(image: np.ndarray, detections: list[dict], panel: bool = True) -> np.ndarray:
    """画检测框 + 类别 + 置信度 + 距离。"""
    items: list[dict] = []
    for det in detections or []:
        box = det.get("bbox") or det.get("box")
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        label = str(det.get("class_name") or det.get("label") or "object")
        score = det.get("score")
        text = f"{label} {score:.2f}" if isinstance(score, (int, float)) else label
        ranging = det.get("ranging") or {}
        if isinstance(ranging.get("depth_m"), (int, float)):
            text += f" {ranging['depth_m']:.2f}m"
        cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_DETECT, 2)
        if panel:
            items.append({"text": text, "xy": (x1 + 4, max(0, y1 - 20)), "color": (25, 25, 25),
                          "size": 15, "bg": COLOR_DETECT, "padding": 3})
    return draw_text_items(image, items)


def draw_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35,
              color=COLOR_SEG, draw_contour: bool = True) -> np.ndarray:
    """半透明掩膜叠加 + 轮廓。"""
    if mask is None:
        return image
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.shape[:2] != image.shape[:2]:
        binary = cv2.resize(binary, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    overlay[binary > 0] = color
    out = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    if draw_contour:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, color, 2)
    return out


def draw_ranging_points(image: np.ndarray, result: dict) -> np.ndarray:
    """画边界采样点：普通绿 / 最近红 / 最远蓝；道路左右用青与品红。"""
    kind = str(result.get("target_kind", "object"))
    points = result.get("sample_points") or []
    if isinstance(points, str) or not points:
        return image

    near_pt = result.get("_nearest_point")
    far_pt = result.get("_farthest_point")
    base_color = {"road_left": COLOR_LEFT, "road_right": COLOR_RIGHT}.get(kind, COLOR_SAMPLE)

    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x, y = int(pt[0]), int(pt[1])
        if near_pt and (x, y) == (int(near_pt[0]), int(near_pt[1])):
            continue
        if far_pt and (x, y) == (int(far_pt[0]), int(far_pt[1])):
            continue
        cv2.circle(image, (x, y), 3, base_color, -1)

    items = []
    depth = result.get("depth_m")
    if near_pt:
        x, y = int(near_pt[0]), int(near_pt[1])
        cv2.circle(image, (x, y), 7, COLOR_MIN, -1)
        cv2.circle(image, (x, y), 11, (255, 255, 255), 2)
        if isinstance(depth, (int, float)):
            items.append({"text": f"Min {depth:.2f}m", "xy": (x + 12, y - 22), "color": COLOR_MIN,
                          "size": 16})
    if far_pt:
        x, y = int(far_pt[0]), int(far_pt[1])
        cv2.circle(image, (x, y), 7, COLOR_MAX, -1)
        cv2.circle(image, (x, y), 11, (255, 255, 255), 2)
        if isinstance(depth, (int, float)):
            items.append({"text": f"Max {depth:.2f}m", "xy": (x + 12, y + 8), "color": COLOR_MAX,
                          "size": 16})
    return draw_text_items(image, items)


def draw_hud(image: np.ndarray, lines: list[str], origin: tuple[int, int] = (12, 10),
             style: HudStyle = HudStyle()) -> np.ndarray:
    items = [{"text": line, "xy": (int(origin[0]), int(origin[1]) + i * style.line_height),
              "color": style.color, "size": style.size} for i, line in enumerate(lines)]
    return draw_text_items(image, items)


def draw_panel(image: np.ndarray, lines: list[str], anchor: str = "top-right",
               title: str | None = None, size: int = 16) -> np.ndarray:
    """在指定角落画信息面板（用于道路左右边界等结构化信息）。"""
    if not lines:
        return image
    h, w = image.shape[:2]
    line_h = size + 10
    bg = (24, 24, 24)
    color = COLOR_TEXT
    title_color = COLOR_TITLE

    widths = [text_size(t, size)[0] for t in lines]
    if title:
        widths.append(text_size(title, size + 2)[0])
    box_w = max(widths) + 28
    box_h = line_h * len(lines) + (line_h + 14 if title else 18)

    if anchor == "top-right":
        x0, y0 = w - box_w - 12, 12
    elif anchor == "bottom-left":
        x0, y0 = 12, h - box_h - 12
    elif anchor == "bottom-right":
        x0, y0 = w - box_w - 12, h - box_h - 12
    else:
        x0, y0 = 12, 12

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), bg, -1)
    out = cv2.addWeighted(overlay, 0.62, image, 0.38, 0)

    items: list[dict] = []
    y = y0 + 10
    if title:
        items.append({"text": title, "xy": (x0 + 14, y), "color": title_color, "size": size + 2})
        y += line_h + 4
    for line in lines:
        items.append({"text": line, "xy": (x0 + 14, y), "color": color, "size": size})
        y += line_h
    return draw_text_items(out, items)


def compose_side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1] + gap
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap:] = right
    return canvas
