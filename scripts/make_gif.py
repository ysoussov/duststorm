import os
import math
import random
from typing import Tuple, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance
import subprocess
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(PROJECT_ROOT, "inputs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def ensure_dirs() -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(path: str) -> Optional[Image.Image]:
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGBA")


def detect_face_region(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if cv2 is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # Pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def extract_head(image_path: str, target_size: int = 160) -> Image.Image:
    """
    Returns a square RGBA head cutout. Falls back to a center crop if face detection fails.
    """
    img = Image.open(image_path).convert("RGBA")
    if cv2 is not None:
        img_bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        box = detect_face_region(img_bgr)
    else:
        box = None

    if box is None:
        # Center crop fallback
        w, h = img.size
        side = min(w, h) // 2
        left = (w - side) // 2
        top = (h - side) // 2
        head = img.crop((left, top, left + side, top + side))
    else:
        x, y, w_box, h_box = box
        pad = int(0.25 * h_box)
        head = img.crop((max(0, x - pad), max(0, y - pad), x + w_box + pad, y + h_box + pad))

    head = head.resize((target_size, target_size), Image.LANCZOS)

    # Make a circular alpha mask for a sticker-like head
    mask = Image.new("L", head.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((2, 2, head.size[0] - 2, head.size[1] - 2), fill=255)
    head.putalpha(mask)
    return head


def placeholder_head(initials: str, color: Tuple[int, int, int] = (60, 120, 240)) -> Image.Image:
    size = 160
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((0, 0, size - 1, size - 1), fill=color + (255,), outline=(255, 255, 255, 255), width=6)
    try:
        font = ImageFont.truetype("Arial.ttf", 56)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), initials, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) / 2, (size - th) / 2), initials, fill=(255, 255, 255, 255), font=font)
    return img


def extract_subject(image_path: str, target_height: int = 340) -> Optional[Image.Image]:
    """Rudimentary automatic subject cutout using GrabCut. Returns RGBA image or None."""
    if cv2 is None:
        return None

    bgr = cv2.imread(image_path)
    if bgr is None:
        return None

    # Limit size for speed
    h, w = bgr.shape[:2]
    scale = 800 / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]

    # Initial rectangle slightly inset
    margin = int(min(h, w) * 0.08)
    rect = (margin, margin, max(1, w - 2 * margin), max(1, h - 2 * margin))
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask2
    pil = Image.fromarray(rgba)

    # Resize by height
    if target_height is not None and target_height > 0:
        ratio = target_height / pil.height
        pil = pil.resize((int(pil.width * ratio), target_height), Image.LANCZOS)
    return pil


def cartoonify_rgba(pil_img: Image.Image) -> Image.Image:
    """Convert an RGBA image into a simple cartoon style while keeping transparency."""
    rgba = pil_img.convert("RGBA")
    if cv2 is None:
        # PIL fallback: posterize + edge enhancement
        base = rgba.convert("RGB")
        # Reduce color levels for flat fill
        base = base.convert("P", palette=Image.ADAPTIVE, colors=64).convert("RGB")
        base = base.filter(ImageFilter.SMOOTH_MORE)
        edges = rgba.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.SMOOTH)
        # Strengthen edges
        edges = edges.point(lambda p: 0 if p < 24 else 255)
        edges_rgb = Image.merge("RGB", (edges, edges, edges))
        color = base
        # Draw black edges
        color = Image.composite(Image.new("RGB", color.size, (0, 0, 0)), color, edges)
        out = Image.merge("RGBA", (*color.split(), rgba.split()[3]))
        return out

    # OpenCV path
    arr = np.array(rgba)
    bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = arr[:, :, 3]

    # Color smoothing via bilateral filtering
    smooth = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

    # Edge detection
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    edges = 255 - edges  # edges as white
    edges = cv2.GaussianBlur(edges, (3, 3), 0)

    # Combine: draw black edges over smoothed color
    color = cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    color = np.where(edges_rgb > 200, 0, color)  # put black where strong edges

    result = np.dstack([color, alpha])
    return Image.fromarray(result, mode="RGBA")


def draw_desert(width: int, height: int) -> Image.Image:
    # Burning Man inspired background (playa, distant mountains, art structures)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Sky with slight haze
    sky = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sky_draw = ImageDraw.Draw(sky)
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(240 * (1 - t) + 255 * t)
        g = int(220 * (1 - t) + 235 * t)
        b = int(190 * (1 - t) + 220 * t)
        sky_draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    # Distant mountains silhouette
    mnt = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    md = ImageDraw.Draw(mnt)
    ridge = []
    for x in range(0, width + 1, 40):
        y = int(height * 0.52 + 20 * math.sin(x * 0.01) + 35 * math.sin(x * 0.03 + 1))
        ridge.append((x, y))
    ridge = [(0, height), (0, ridge[0][1])] + ridge + [(width, ridge[-1][1]), (width, height)]
    md.polygon(ridge, fill=(120, 130, 150, 180))
    mnt = mnt.filter(ImageFilter.GaussianBlur(radius=2))

    # Playa (alkali flats)
    playa = Image.new("RGBA", (width, height // 2), (230, 220, 200, 255))
    # subtle cracks/noise
    pd = ImageDraw.Draw(playa)
    for i in range(140):
        x = random.randint(0, width)
        y = random.randint(0, playa.size[1] - 1)
        ln = random.randint(10, 40)
        ang = random.uniform(-math.pi / 2, math.pi / 2)
        x2 = int(x + ln * math.cos(ang))
        y2 = int(y + ln * math.sin(ang))
        pd.line([(x, y), (x2, y2)], fill=(200, 190, 175, 70), width=1)

    # Composite layers
    img.alpha_composite(sky, (0, 0))
    img.alpha_composite(mnt, (0, 0))
    img.alpha_composite(playa, (0, height - playa.size[1]))

    # Simple temple/man silhouette and a dome
    d = ImageDraw.Draw(img)
    # The Man on a base
    base_y = int(height * 0.68)
    cx = int(width * 0.38)
    d.rectangle((cx - 18, base_y - 12, cx + 18, base_y), fill=(90, 90, 90, 200))
    # body
    d.line([(cx, base_y - 12), (cx, base_y - 80)], fill=(100, 100, 100, 220), width=5)
    # arms
    d.line([(cx - 30, base_y - 55), (cx + 30, base_y - 55)], fill=(100, 100, 100, 220), width=5)
    # legs
    d.line([(cx, base_y - 12), (cx - 20, base_y - 12 - 35)], fill=(100, 100, 100, 220), width=5)
    d.line([(cx, base_y - 12), (cx + 20, base_y - 12 - 35)], fill=(100, 100, 100, 220), width=5)
    # head
    d.ellipse((cx - 8, base_y - 92, cx + 8, base_y - 76), fill=(120, 120, 120, 220))

    # Geodesic dome
    dome_c = (int(width * 0.7), base_y - 10)
    dome_r = 60
    d.ellipse((dome_c[0] - dome_r, dome_c[1] - dome_r, dome_c[0] + dome_r, dome_c[1] + dome_r), outline=(110, 110, 120, 220), width=3)
    for a in range(0, 180, 20):
        x = dome_c[0] + int(dome_r * math.cos(math.radians(a)))
        y = dome_c[1] - int(dome_r * math.sin(math.radians(a)))
        d.line([dome_c, (x, y)], fill=(110, 110, 120, 160), width=2)
    # art car
    car_y = base_y + 8
    d.rectangle((int(width * 0.15), car_y - 16, int(width * 0.23), car_y), fill=(120, 120, 120, 180))
    d.ellipse((int(width * 0.16), car_y, int(width * 0.18), car_y + 18), fill=(80, 80, 80, 220))
    d.ellipse((int(width * 0.20), car_y, int(width * 0.22), car_y + 18), fill=(80, 80, 80, 220))
    d.line([(int(width * 0.215), car_y - 16), (int(width * 0.215), car_y - 48)], fill=(120, 120, 120, 180), width=3)
    d.polygon([(int(width * 0.215), car_y - 48), (int(width * 0.195), car_y - 40), (int(width * 0.215), car_y - 32)], fill=(220, 50, 50, 200))

    return img


def draw_cactus(canvas: Image.Image, x: int, base_y: int, scale: float = 1.0) -> None:
    draw = ImageDraw.Draw(canvas)
    trunk_w = int(50 * scale)
    trunk_h = int(220 * scale)
    trunk_rect = [x, base_y - trunk_h, x + trunk_w, base_y]
    green = (44, 160, 83, 255)
    dark = (24, 100, 50, 255)
    draw.rounded_rectangle(trunk_rect, radius=int(18 * scale), fill=green, outline=dark, width=4)

    def arm(cx: int, cy: int, w: int, h: int, flip: int) -> None:
        rect = [cx - w, cy - h, cx + w, cy + h]
        draw.rounded_rectangle(rect, radius=int(w * 0.9), fill=green, outline=dark, width=4)
        # little tip
        tip = [cx - w, cy - h - w, cx + w, cy - h + w]
        draw.ellipse(tip, fill=green, outline=dark)

    arm(x - int(10 * scale), base_y - int(120 * scale), int(20 * scale), int(70 * scale), -1)
    arm(x + trunk_w + int(10 * scale), base_y - int(150 * scale), int(20 * scale), int(90 * scale), 1)

    # Spines
    for i in range(18):
        px = x + random.randint(6, trunk_w - 6)
        py = base_y - random.randint(10, trunk_h - 10)
        length = random.randint(6, 12)
        angle = random.uniform(-math.pi, math.pi)
        qx = int(px + length * math.cos(angle))
        qy = int(py + length * math.sin(angle))
        draw.line([(px, py), (qx, qy)], fill=(230, 230, 230, 255), width=2)


def make_dust_layer(width: int, height: int, strength: float, angle_deg: float) -> Image.Image:
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    num_particles = int(200 * strength) + 80
    for _ in range(num_particles):
        x = random.randint(-20, width + 20)
        y = random.randint(0, height - 1)
        sz = random.randint(2, 4)
        draw.ellipse((x, y, x + sz, y + sz), fill=(200, 170, 120, random.randint(80, 160)))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=2))
    layer = layer.rotate(angle_deg, resample=Image.BICUBIC, expand=0)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=1))
    return layer


def add_text(canvas: Image.Image, text: str, xy: Tuple[int, int], color=(255, 255, 255, 230)) -> None:
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    x, y = xy
    draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 160), font=font)
    draw.text((x, y), text, fill=color, font=font)


def add_speed_lines(canvas: Image.Image, center: Tuple[int, int], angle_deg: float, length: int, count: int = 18, color=(255, 255, 255, 110)) -> None:
    draw = ImageDraw.Draw(canvas)
    angle = math.radians(angle_deg)
    cx, cy = center
    for i in range(count):
        spread = (i - count / 2) / (count / 2)
        offset = int(spread * 40)
        x0 = cx - int(length * math.cos(angle)) + offset
        y0 = cy - int(length * math.sin(angle)) + offset
        x1 = cx + int(length * 0.2 * math.cos(angle)) + offset
        y1 = cy + int(length * 0.2 * math.sin(angle)) + offset
        draw.line([(x0, y0), (x1, y1)], fill=color, width=2)


def draw_speech_bubble(canvas: Image.Image, text: str, anchor: Tuple[int, int], pointing_to: Tuple[int, int], bubble_color=(255, 255, 255, 240), outline=(30, 30, 30, 255)) -> None:
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    margin = 16
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    w = tw + margin * 2
    h = th + margin * 2
    x, y = anchor
    rect = (x, y, x + w, y + h)
    draw.rounded_rectangle(rect, radius=18, fill=bubble_color, outline=outline, width=4)
    # tail
    px, py = pointing_to
    tail = [(x + w - 30, y + h), (x + w - 10, y + h), (px, py)]
    draw.polygon(tail, fill=bubble_color, outline=outline)
    draw.text((x + margin, y + margin - 4), text, fill=(20, 20, 20, 255), font=font)


def draw_starburst(canvas: Image.Image, center: Tuple[int, int], radius: int, spikes: int = 14, color=(255, 230, 120, 240), outline=(120, 60, 0, 255)) -> None:
    draw = ImageDraw.Draw(canvas)
    cx, cy = center
    pts = []
    for i in range(spikes * 2):
        r = radius if i % 2 == 0 else int(radius * 0.55)
        a = (i / (spikes * 2)) * 2 * math.pi
        pts.append((cx + int(r * math.cos(a)), cy + int(r * math.sin(a))))
    draw.polygon(pts, fill=color, outline=outline)


def apply_shake_and_zoom(frame: Image.Image, shake_px: int, zoom: float) -> Image.Image:
    if zoom != 1.0:
        w, h = frame.size
        zw, zh = int(w * zoom), int(h * zoom)
        fr = frame.resize((zw, zh), Image.LANCZOS)
        # crop center back to original size
        left = (zw - w) // 2
        top = (zh - h) // 2
        frame = fr.crop((left, top, left + w, top + h))
    if shake_px > 0:
        dx = random.randint(-shake_px, shake_px)
        dy = random.randint(-shake_px, shake_px)
        shaken = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        shaken.alpha_composite(frame, (dx, dy))
        return shaken
    return frame


def apply_color_grade(frame: Image.Image, warmth: float = 1.08, contrast: float = 1.12, brightness: float = 1.03) -> Image.Image:
    graded = frame
    # Subtle warm tint using overlay blend
    warm_layer = Image.new("RGBA", graded.size, (255, 200, 120, 40))
    graded = Image.alpha_composite(graded, warm_layer)
    # Enhance contrast and brightness
    graded = ImageEnhance.Contrast(graded).enhance(contrast)
    graded = ImageEnhance.Brightness(graded).enhance(brightness)
    graded = ImageEnhance.Color(graded).enhance(warmth)
    return graded


def draw_lightning(canvas: Image.Image, start: Tuple[int, int], end: Tuple[int, int], forks: int = 2) -> None:
    draw = ImageDraw.Draw(canvas)
    points = [start]
    steps = 6
    for i in range(1, steps):
        t = i / steps
        jitter = 30 * (1 - t)
        x = int(start[0] + (end[0] - start[0]) * t + random.uniform(-jitter, jitter))
        y = int(start[1] + (end[1] - start[1]) * t + random.uniform(-jitter, jitter))
        points.append((x, y))
    points.append(end)
    # Glow underlay
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.line(points, fill=(255, 255, 180, 220), width=6)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=6))
    canvas.alpha_composite(glow)
    # Core bolt
    draw.line(points, fill=(255, 255, 255, 255), width=3)
    # Small forks near the end
    for k in range(forks):
        if len(points) < 3:
            break
        base = points[-(2 + k)]
        fork_end = (base[0] + random.randint(-50, 50), base[1] + random.randint(-60, -20))
        draw.line([base, fork_end], fill=(255, 255, 255, 220), width=2)


def compose_frames(img1: Image.Image, img2: Image.Image, width: int = 960, height: int = 720, frames: int = 240) -> list[Image.Image]:
    rng = random.Random(42)
    result = []
    base = draw_desert(width, height)
    cactus_x = int(width * 0.22)
    ground_y = int(height * 0.85)

    for i in range(frames):
        t = i / max(1, frames - 1)
        frame = base.copy()

        # Wind strength ramps up
        wind = 0.4 + 0.9 * t
        angle = -15 - 10 * math.sin(t * math.pi)
        dust = make_dust_layer(width, height, strength=wind, angle_deg=angle)

        # Person 1 clinging wobble
        wobble = math.sin(i * 0.4) * 8
        h1, w1 = img1.height, img1.width
        # place person1 on the right side of the cactus
        p1x = cactus_x + 40
        p1y = ground_y - h1 - 10 + int(wobble)
        img1_rot = img1.rotate(wobble * 0.6 - 6, resample=Image.BICUBIC)
        # Draw person1 first (behind cactus)
        frame.alpha_composite(img1_rot, (p1x, p1y))

        # Draw cactus on top so person appears to cling behind it
        draw_cactus(frame, cactus_x, ground_y, scale=1.0)

        # Person 2 enters from right to save starting mid-way
        if t < 0.25:
            # not yet
            pass
        else:
            enter = min(1.0, (t - 0.25) / 0.55)
            h2, w2 = img2.height, img2.width
            fly_bob = int(math.sin(i * 0.45) * 6)
            # Target is near person1's rescue hand
            target_x = p1x + int(img1.width * 0.82)
            target_y = p1y + int(img1.height * 0.60)
            start_x = width + w2
            start_y = int(height * 0.25)
            arc_h = int(height * 0.25)
            ease = 1 - pow(1 - enter, 3)  # ease-out cubic
            p2x = int(start_x + (target_x - start_x) * ease)
            p2y = int(start_y + (target_y - start_y) * ease - arc_h * math.sin(math.pi * ease)) + fly_bob

            # Draw cape between cactus and hero (enhanced, with shading and flutter)
            cape_draw = ImageDraw.Draw(frame)
            base_color = (220, 30, 50, 220)
            shadow_color = (150, 20, 40, 180)
            highlight = (255, 90, 110, 180)
            wind_amp = int(90 + 60 * math.sin(i * 0.35 + 1.2) + 30 * wind)
            cape_attach = (p2x + int(w2 * 0.55), p2y + int(h2 * 0.18))
            wave1 = (cape_attach[0] - wind_amp, cape_attach[1] + int(h2 * 0.25))
            wave2 = (cape_attach[0] - int(wind_amp * 0.6), cape_attach[1] + int(h2 * 0.55))
            wave3 = (cape_attach[0] - int(wind_amp * 1.2), cape_attach[1] + int(h2 * 0.42))
            poly = [
                (cape_attach[0], cape_attach[1] - 6),
                (cape_attach[0] + 8, cape_attach[1] + 2),
                wave1,
                wave2,
                wave3,
                (cape_attach[0] - 12, cape_attach[1] + 12),
            ]
            cape_draw.polygon(poly, fill=base_color)
            # Shadow fold
            cape_draw.line([cape_attach, wave2], fill=shadow_color, width=16)
            # Highlight edge
            cape_draw.line([cape_attach, wave3], fill=highlight, width=6)

            img2_rot = img2.rotate(-8, resample=Image.BICUBIC)
            # Rescuer in front of cactus
            frame.alpha_composite(img2_rot, (p2x, p2y))

            # Emblem and mask on top of hero (customizable)
            deco = ImageDraw.Draw(frame)
            emblem_letter = os.environ.get('HERO_EMBLEM', 'H')[:1]
            emblem_fill = tuple(int(x) for x in os.environ.get('HERO_EMBLEM_COLOR', '255,215,0').split(',')) + (240,)
            emblem_icon = tuple(int(x) for x in os.environ.get('HERO_ICON_COLOR', '255,60,60').split(',')) + (255,)
            mask_color = tuple(int(x) for x in os.environ.get('HERO_MASK_COLOR', '0,0,0').split(',')) + (220,)

            chest_cx = p2x + int(w2 * 0.55)
            chest_cy = p2y + int(h2 * 0.45)
            deco.ellipse((chest_cx - 22, chest_cy - 22, chest_cx + 22, chest_cy + 22), fill=emblem_fill, outline=(120, 60, 0, 255), width=3)
            deco.regular_polygon((chest_cx, chest_cy, 12), n_sides=5, rotation=18, fill=emblem_icon)
            try:
                font = ImageFont.truetype("Arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            tw, th = deco.textbbox((0, 0), emblem_letter, font=font)[2:]
            deco.text((chest_cx - tw / 2, chest_cy - th / 2), emblem_letter, fill=(20, 20, 20, 255), font=font)
            # Eye mask
            eye_y = p2y + int(h2 * 0.22)
            deco.rounded_rectangle((p2x + int(w2 * 0.42), eye_y - 6, p2x + int(w2 * 0.68), eye_y + 10), radius=6, fill=mask_color)

            # Draw a simple connecting arm cartoon line (rescue grip)
            hand1 = (p1x + int(img1.width * 0.82), p1y + int(img1.height * 0.60))
            hand2 = (p2x + int(w2 * 0.20), p2y + int(h2 * 0.62))
            deco.line([hand1, hand2], fill=(80, 50, 30, 255), width=10)

            # Speed lines around hero
            add_speed_lines(frame, (p2x + int(w2 * 0.5), p2y + int(h2 * 0.4)), angle_deg=-20, length=180, count=22, color=(255, 255, 255, 140))

        # Ensure visible grip around the cactus: palm + fingers in front
        grip_draw = ImageDraw.Draw(frame)
        trunk_w = 50
        grip_y = ground_y - 210 + int(wobble)
        skin = (255, 220, 185, 255)
        outline = (70, 45, 30, 255)
        # Fingers in front of trunk
        for fidx in range(4):
            fx0 = cactus_x + 6 + fidx * int((trunk_w - 10) / 4)
            fx1 = fx0 + int((trunk_w - 30) / 4)
            fy0 = grip_y - 2
            fy1 = grip_y + 10
            grip_draw.rounded_rectangle([fx0, fy0, fx1, fy1], radius=5, fill=skin, outline=outline, width=2)
        # Thumb wrapping around the right edge
        grip_draw.rounded_rectangle([cactus_x + trunk_w - 8, grip_y, cactus_x + trunk_w + 10, grip_y + 18], radius=8, fill=skin, outline=outline, width=2)

        # Foreground dust and occasional big dust burst
        frame.alpha_composite(dust)
        if 0.3 < t < 0.8 and i % 3 == 0:
            burst = make_dust_layer(width, height, strength=1.2, angle_deg=angle - 10)
            frame.alpha_composite(burst)

        if i < frames // 2:
            draw_speech_bubble(frame, "Hang on!", (cactus_x + 140, ground_y - 340), (cactus_x + 70, ground_y - 230))
        else:
            draw_speech_bubble(frame, "I got you!", (cactus_x + 180, ground_y - 360), (cactus_x + 90, ground_y - 260))
        if 0.3 < t < 0.65 and i % 4 == 0:
            add_text(frame, "WHOOOOSH!", (int(width * 0.55), int(height * 0.18)), color=(255, 230, 120, 230))
        if abs(t - 0.7) < 0.02:
            draw_starburst(frame, (int(width * 0.46), int(height * 0.28)), radius=90)
            add_text(frame, "THWUMP!", (int(width * 0.42), int(height * 0.25)), color=(60, 20, 20, 255))

        # Vignette
        vignette = Image.new("L", (width, height), 0)
        dv = ImageDraw.Draw(vignette)
        dv.ellipse((-int(width * 0.2), -int(height * 0.2), int(width * 1.2), int(height * 1.2)), fill=255)
        vignette = vignette.filter(ImageFilter.GaussianBlur(radius=80))
        frame.putalpha(255)
        frame = Image.composite(frame, Image.new("RGBA", (width, height), (0, 0, 0, 255)), vignette)

        # Camera shake and dramatic zoom near rescue
        zoom = 1.0 + 0.08 * (1 if t > 0.5 else t * 2)
        shake = int(2 + 8 * wind)
        frame = apply_shake_and_zoom(frame, shake_px=shake, zoom=zoom)

        # Color grade for punch
        frame = apply_color_grade(frame, warmth=1.1, contrast=1.18, brightness=1.04)

        # Occasional lightning flash in clouds
        if 0.55 < t < 0.65 and i % 8 == 0:
            draw_lightning(frame, (int(width * 0.75), int(height * 0.1)), (int(width * 0.6), int(height * 0.35)))

        # Duplicate a couple frames around impact for a tiny hold
        result.append(frame)
        if abs(t - 0.7) < 0.02:
            result.append(frame.copy())
            result.append(frame.copy())
    return result


def save_gif(frames: list[Image.Image], path: str, fps: int = 10) -> None:
    duration = int(1000 / fps)
    # Build a global adaptive palette from the first frame (RGB) to preserve colors
    base_rgb = frames[0].convert("RGB")
    first_p = base_rgb.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
    pal = first_p.getpalette()
    pal_img = Image.new('P', (1, 1))
    pal_img.putpalette(pal)

    pal_frames = [
        f.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG, palette=pal_img)
        for f in frames
    ]

    pal_frames[0].save(
        path,
        save_all=True,
        append_images=pal_frames[1:],
        duration=duration,
        loop=0,
        disposal=2,
        optimize=False,
    )


def main() -> None:
    ensure_dirs()
    person1_path = os.path.join(INPUT_DIR, "person1.jpg")
    person2_path = os.path.join(INPUT_DIR, "person2.jpg")

    # Prefer full-subject cutouts for realism, fall back to circular heads
    if os.path.exists(person1_path):
        img1 = extract_subject(person1_path, target_height=360) or extract_head(person1_path)
    else:
        img1 = placeholder_head("P1", (60, 140, 255))

    if os.path.exists(person2_path):
        img2 = extract_subject(person2_path, target_height=360) or extract_head(person2_path)
    else:
        img2 = placeholder_head("P2", (255, 120, 60))

    frames = compose_frames(img1, img2)
    # Save MP4 using imageio-ffmpeg
    mp4_path = os.path.join(OUTPUT_DIR, "duststorm.mp4")
    fps = 24
    writer = imageio.get_writer(mp4_path, fps=fps, codec='libx264', quality=8, macro_block_size=None)
    try:
        for f in frames:
            writer.append_data(np.array(f.convert("RGB")))
    finally:
        writer.close()
    print(f"Saved: {mp4_path}")

    # Generate or use custom audio SFX and mux
    out_with_audio = os.path.join(OUTPUT_DIR, "duststorm_with_audio.mp4")
    duration = 10

    inputs_dir = INPUT_DIR
    custom_wind = os.environ.get('WIND_SFX', os.path.join(inputs_dir, 'wind.wav'))
    custom_whoosh = os.environ.get('WHOOSH_SFX', os.path.join(inputs_dir, 'whoosh.wav'))
    custom_impact = os.environ.get('IMPACT_SFX', os.path.join(inputs_dir, 'impact.wav'))

    wind_offset = float(os.environ.get('WIND_OFFSET', '0.0'))
    whoosh_offset = float(os.environ.get('WHOOSH_OFFSET', '3.0'))
    impact_offset = float(os.environ.get('IMPACT_OFFSET', '7.0'))

    wind_vol = float(os.environ.get('WIND_VOL', '0.8'))
    whoosh_vol = float(os.environ.get('WHOOSH_VOL', '1.1'))
    impact_vol = float(os.environ.get('IMPACT_VOL', '1.6'))

    wind_path = custom_wind if os.path.exists(custom_wind) else os.path.join(OUTPUT_DIR, 'wind.wav')
    whoosh_path = custom_whoosh if os.path.exists(custom_whoosh) else os.path.join(OUTPUT_DIR, 'whoosh.wav')
    impact_path = custom_impact if os.path.exists(custom_impact) else os.path.join(OUTPUT_DIR, 'impact.wav')

    # Synthesize defaults only if not provided
    if not os.path.exists(custom_wind):
        subprocess.run(['ffmpeg','-y','-f','lavfi','-i','anoisesrc=color=pink:amplitude=0.35','-t',str(duration), wind_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(custom_whoosh):
        subprocess.run(['ffmpeg','-y','-f','lavfi','-i','sine=f=220:duration=0.6','-af','asetrate=44100*1.7,atempo=1.0,alimiter=level_in=5:level_out=1', whoosh_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(custom_impact):
        subprocess.run(['ffmpeg','-y','-f','lavfi','-i','sine=f=55:duration=0.25','-af','aecho=0.7:0.88:12:0.35,alimiter=level_in=3:level_out=1', impact_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Volume-adjust temporary files
    wind_adj = os.path.join(OUTPUT_DIR, 'wind_adj.wav')
    whoosh_adj = os.path.join(OUTPUT_DIR, 'whoosh_adj.wav')
    impact_adj = os.path.join(OUTPUT_DIR, 'impact_adj.wav')
    subprocess.run(['ffmpeg','-y','-i', wind_path,'-filter:a', f'volume={wind_vol}', wind_adj], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['ffmpeg','-y','-i', whoosh_path,'-filter:a', f'volume={whoosh_vol}', whoosh_adj], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['ffmpeg','-y','-i', impact_path,'-filter:a', f'volume={impact_vol}', impact_adj], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Mix with offsets
    mix_aud = os.path.join(OUTPUT_DIR, 'mix.wav')
    subprocess.run([
        'ffmpeg','-y',
        '-i', wind_adj,
        '-itsoffset', str(whoosh_offset), '-i', whoosh_adj,
        '-itsoffset', str(impact_offset), '-i', impact_adj,
        '-filter_complex','[0:a][1:a][2:a]amix=inputs=3:dropout_transition=0:weights=1 1 1,volume=1.1',
        '-t', str(duration), mix_aud
    ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Mux
    subprocess.run(['ffmpeg','-y','-i', mp4_path,'-i', mix_aud,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-shortest', out_with_audio], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved: {out_with_audio}")


if __name__ == "__main__":
    main()


