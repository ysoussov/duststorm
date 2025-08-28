import os
import math
import random
from typing import Tuple, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import imageio.v2 as imageio

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
    # Sky gradient
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sky = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sky_draw = ImageDraw.Draw(sky)
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(255 * (1 - t) + 255 * t)
        g = int(200 * (1 - t) + 215 * t)
        b = int(120 * (1 - t) + 170 * t)
        sky_draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
    # Sand
    sand = Image.new("RGBA", (width, height // 3), (225, 190, 120, 255))
    img.alpha_composite(sky, (0, 0))
    img.alpha_composite(sand, (0, height - height // 3))
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


def compose_frames(img1: Image.Image, img2: Image.Image, width: int = 960, height: int = 720, frames: int = 40) -> list[Image.Image]:
    rng = random.Random(42)
    result = []
    base = draw_desert(width, height)
    cactus_x = int(width * 0.22)
    ground_y = int(height * 0.85)

    for i in range(frames):
        t = i / max(1, frames - 1)
        frame = base.copy()

        # Wind strength ramps up
        wind = 0.3 + 0.7 * t
        angle = -15 - 10 * math.sin(t * math.pi)
        dust = make_dust_layer(width, height, strength=wind, angle_deg=angle)

        # Person 1 clinging wobble
        wobble = math.sin(i * 0.4) * 8
        h1, w1 = img1.height, img1.width
        p1x = cactus_x - w1 + 20
        p1y = ground_y - h1 - 10 + int(wobble)
        img1_rot = img1.rotate(wobble * 0.6 - 6, resample=Image.BICUBIC)
        # Draw person1 first (behind cactus)
        frame.alpha_composite(img1_rot, (p1x, p1y))

        # Draw cactus on top so person appears to cling behind it
        draw_cactus(frame, cactus_x, ground_y, scale=1.0)

        # Person 2 enters from right to save starting mid-way
        if t < 0.45:
            # not yet
            pass
        else:
            enter = min(1.0, (t - 0.45) / 0.45)
            h2, w2 = img2.height, img2.width
            fly_bob = int(math.sin(i * 0.45) * 6)
            p2x = int(width - w2 - 40 - 420 * (1 - enter))
            p2y = ground_y - h2 - 60 + fly_bob  # hover slightly for superhero vibe

            # Draw cape between cactus and hero
            cape_draw = ImageDraw.Draw(frame)
            cape_color = (220, 30, 50, 220)
            cape_wind = int(80 + 40 * math.sin(i * 0.3 + 1.0))
            cape_attach = (p2x + int(w2 * 0.55), p2y + int(h2 * 0.18))
            cape_tip1 = (cape_attach[0] - cape_wind, cape_attach[1] + int(h2 * 0.25))
            cape_tip2 = (cape_attach[0] - cape_wind // 2, cape_attach[1] + int(h2 * 0.55))
            cape_poly = [
                (cape_attach[0], cape_attach[1] - 6),
                (cape_attach[0] + 6, cape_attach[1] + 2),
                cape_tip1,
                cape_tip2,
                (cape_attach[0] - 10, cape_attach[1] + 12),
            ]
            cape_draw.polygon(cape_poly, fill=cape_color)

            img2_rot = img2.rotate(-8, resample=Image.BICUBIC)
            # Rescuer in front of cactus
            frame.alpha_composite(img2_rot, (p2x, p2y))

            # Emblem and mask on top of hero
            deco = ImageDraw.Draw(frame)
            chest_cx = p2x + int(w2 * 0.55)
            chest_cy = p2y + int(h2 * 0.45)
            deco.ellipse((chest_cx - 22, chest_cy - 22, chest_cx + 22, chest_cy + 22), fill=(255, 215, 0, 240), outline=(120, 60, 0, 255), width=3)
            deco.regular_polygon((chest_cx, chest_cy, 12), n_sides=5, rotation=18, fill=(255, 60, 60, 255))
            # Eye mask
            eye_y = p2y + int(h2 * 0.22)
            deco.rounded_rectangle((p2x + int(w2 * 0.42), eye_y - 6, p2x + int(w2 * 0.68), eye_y + 10), radius=6, fill=(0, 0, 0, 220))

            # Draw a simple connecting arm cartoon line (rescue grip)
            hand1 = (p1x + int(img1.width * 0.82), p1y + int(img1.height * 0.60))
            hand2 = (p2x + int(w2 * 0.20), p2y + int(h2 * 0.62))
            deco.line([hand1, hand2], fill=(80, 50, 30, 255), width=10)

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
        # Thumb wrapping around the left edge
        grip_draw.rounded_rectangle([cactus_x - 10, grip_y, cactus_x + 8, grip_y + 18], radius=8, fill=skin, outline=outline, width=2)

        # Foreground dust
        frame.alpha_composite(dust)

        if i < frames // 2:
            add_text(frame, "Hang on!", (cactus_x + 80, ground_y - 300))
        else:
            add_text(frame, "I got you!", (cactus_x + 120, ground_y - 320))

        # Vignette
        vignette = Image.new("L", (width, height), 0)
        dv = ImageDraw.Draw(vignette)
        dv.ellipse((-int(width * 0.2), -int(height * 0.2), int(width * 1.2), int(height * 1.2)), fill=255)
        vignette = vignette.filter(ImageFilter.GaussianBlur(radius=80))
        frame.putalpha(255)
        frame = Image.composite(frame, Image.new("RGBA", (width, height), (0, 0, 0, 255)), vignette)

        result.append(frame)
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
    out_path = os.path.join(OUTPUT_DIR, "duststorm.gif")
    save_gif(frames, out_path, fps=12)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


