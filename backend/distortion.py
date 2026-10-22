"""
Multi-Mode Distortion Engine for Vision Duel.
Supports multiple distinct, fun corruption styles:
1. 'blur': Gaussian & Defocus Blur
2. 'pixelate': Retro 8-bit Mosaic
3. 'noise': TV Static / Film Grain
4. 'tiles': Mystery Tile / Cutout Grid Reveal
5. 'contrast': High-Contrast Negative/Duotone Fade

Progress curve is tuned so:
- Early (0.0): Shapes and silhouettes are hinted but disguised
- Mid (0.5): Key contours and primary colors emerge
- Late (0.85+): Completely crisp and clear, well before timer expiration!
"""
import random
import io
import base64
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

DISTORTION_MODES = [
    {"id": "pixelate", "name": "8-Bit Retro Mosaic", "icon": "👾"},
    {"id": "blur", "name": "Foggy Defocus Blur", "icon": "🌫️"},
    {"id": "noise", "name": "TV Static Noise", "icon": "📺"},
    {"id": "tiles", "name": "Mystery Cutout Grid", "icon": "🧩"},
    {"id": "contrast", "name": "Contrast Silhouette", "icon": "🌓"}
]

def apply_blur_mode(image: Image.Image, progress: float) -> Image.Image:
    """Soft Gaussian blur that clears up cleanly by progress=0.85"""
    if progress >= 0.85:
        return image
    # Radius goes from 9.0px down to 0.0px
    intensity = (1.0 - (progress / 0.85)) ** 1.6
    radius = max(0.0, 9.0 * intensity)
    if radius < 0.3:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_pixelate_mode(image: Image.Image, progress: float) -> Image.Image:
    """Downsamples and rescales with nearest neighbor (mosaic effect)"""
    if progress >= 0.85:
        return image
    orig_w, orig_h = image.size
    # Factor goes from 0.08 (chunky blocks) to 1.0 (full resolution)
    norm = min(1.0, progress / 0.85)
    factor = 0.08 + 0.92 * (norm ** 2.0)
    small_w = max(16, int(orig_w * factor))
    small_h = max(16, int(orig_h * factor))
    small = image.resize((small_w, small_h), resample=Image.Resampling.BILINEAR)
    return small.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)

def apply_noise_mode(image: Image.Image, progress: float, seed: int = 42) -> Image.Image:
    """Adds TV static/grain noise that dissolves into crisp photo.

    Seeded so that the same (image, progress, seed) always yields the same
    frame -- the human and the AI must be looking at identical pixels.
    """
    if progress >= 0.85:
        return image
    intensity = (1.0 - (progress / 0.85)) ** 1.5
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 75 * intensity, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr)
    # Slight blur to blend grain
    if intensity > 0.3:
        noisy_img = noisy_img.filter(ImageFilter.GaussianBlur(radius=1.2 * intensity))
    return noisy_img

def apply_tiles_mode(image: Image.Image, progress: float, seed: int = 42) -> Image.Image:
    """Grid of mystery black tiles that peel away one by one"""
    if progress >= 0.85:
        return image
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    grid_size = 5 # 5x5 = 25 tiles
    tile_w = w / grid_size
    tile_h = h / grid_size

    # Deterministic shuffle of tile positions based on seed
    rng = random.Random(seed)
    all_tiles = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    rng.shuffle(all_tiles)

    # At progress 0: 75% tiles covered. At progress 0.85: 0% covered
    norm = min(1.0, progress / 0.85)
    tiles_to_cover = int(len(all_tiles) * 0.75 * (1.0 - norm))

    for i in range(tiles_to_cover):
        r, c = all_tiles[i]
        x0, y0 = c * tile_w, r * tile_h
        x1, y1 = (c + 1) * tile_w, (r + 1) * tile_h
        draw.rectangle([x0, y0, x1, y1], fill=(15, 23, 42)) # Slate dark tile

    return img_copy

def apply_contrast_mode(image: Image.Image, progress: float) -> Image.Image:
    """Extreme contrast silhouette that restores natural tones"""
    if progress >= 0.85:
        return image
    norm = min(1.0, progress / 0.85)
    intensity = 1.0 - norm

    # Contrast boost (up to 3.0x) and desaturation
    enhancer_con = ImageEnhance.Contrast(image)
    high_con = enhancer_con.enhance(1.0 + 2.2 * intensity)
    
    enhancer_color = ImageEnhance.Color(high_con)
    result = enhancer_color.enhance(1.0 - 0.7 * intensity)
    
    if intensity > 0.2:
        result = result.filter(ImageFilter.GaussianBlur(radius=1.8 * intensity))
    return result

def apply_distortion_by_mode(image: Image.Image, mode: str, progress: float, seed: int = 42) -> Image.Image:
    """Dispatches to the selected distortion mode."""
    if progress >= 0.85:
        return image

    if mode == "pixelate":
        return apply_pixelate_mode(image, progress)
    elif mode == "noise":
        return apply_noise_mode(image, progress, seed)
    elif mode == "tiles":
        return apply_tiles_mode(image, progress, seed)
    elif mode == "contrast":
        return apply_contrast_mode(image, progress)
    else: # default to blur
        return apply_blur_mode(image, progress)

def frame_at(
    image: Image.Image,
    mode: str,
    t: float,
    time_limit: float,
    seed: int,
) -> Image.Image:
    """The exact frame a player is looking at `t` seconds into a round.

    Single source of truth for the reveal schedule. Both the human's
    pre-rendered frames and the AI's glances go through this function with the
    same round seed, so the two sides always see identical pixels at identical
    moments.
    """
    progress = 0.0 if time_limit <= 0 else max(0.0, min(1.0, t / time_limit))
    return apply_distortion_by_mode(image, mode, progress=progress, seed=seed)


def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Converts a PIL image to a base64 data URI string."""
    buffered = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"
