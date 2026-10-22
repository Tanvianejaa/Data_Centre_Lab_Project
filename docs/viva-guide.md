# 🎓 Vision Duel — Complete Viva Guide

> Written for someone brand-new to the project. Every concept is explained from scratch.

---

## Table of Contents

1. [What the Project Is (Plain English)](#1-what-the-project-is-plain-english)
2. [System Architecture — How the Pieces Fit](#2-system-architecture--how-the-pieces-fit)
3. [Dataset: Why Caltech-101?](#3-dataset-why-caltech-101)
4. [The AI Model: CLIP ViT-B/32 — Explained](#4-the-ai-model-clip-vit-b32--explained)
5. [Difficulty Calibration — The Three Honest Handicaps](#5-difficulty-calibration--the-three-honest-handicaps)
6. [The Five Corruption Modes — Code Explained Line by Line](#6-the-five-corruption-modes--code-explained-line-by-line)
7. [ai_engine.py — Line by Line](#7-ai_enginepy--line-by-line)
8. [dataset.py — Line by Line](#8-datasetpy--line-by-line)
9. [server.py — Line by Line](#9-serverpy--line-by-line)
10. [main.py — Line by Line](#10-mainpy--line-by-line)
11. [Scripts Explained](#11-scripts-explained)
12. [Viva Question Bank](#12-viva-question-bank)

---

## 1. What the Project Is (Plain English)

**Vision Duel** is a web game where you race an AI to identify a blurry/corrupted photo. The image starts heavily distorted and gradually clears up over a 3.5-second countdown. You press a key to lock your answer. The AI does the same — but using a real neural network.

**The scientific question it demonstrates:** Humans are surprisingly good at recognising objects even when images are heavily corrupted. Neural networks, on the other hand, fall apart dramatically under the same distortions. This was formally shown in two real research papers:

- **Hendrycks & Dietterich, ICLR 2019** — showed that standard ImageNet-trained CNNs fail catastrophically on corrupted images
- **Geirhos et al., NeurIPS 2018** — showed that humans rely on object *shape* while CNNs rely heavily on *texture*, making CNNs brittle to corruptions that preserve shape but destroy texture

The game is playable *because* of that gap — at Easy difficulty the AI only gets ~39% correct, so you can actually beat it.

---

## 2. System Architecture — How the Pieces Fit

```
Browser (React/Vite)
        │
        │  HTTP (fetch)
        ▼
FastAPI Server  (main.py → server.py)   port 8000
        │
        ├── dataset.py       picks the image, builds answer choices
        ├── distortion.py    corrupts the image, renders frames
        └── ai_engine.py     CLIP model plays the round, returns decision
```

**Request flow for one round:**

1. Browser sends `POST /api/round/next`
2. Server picks a random image from Caltech-101 (`dataset.py`)
3. Server picks a random corruption mode (`distortion.py`)
4. Server runs the AI against the same corrupted frames (`ai_engine.py`)
5. Server sends back 4 pre-rendered image frames (as base64 JPEG) + AI's decision time
6. Browser shows the frames one by one as the countdown runs
7. Human presses a key → browser sends `POST /api/round/submit`
8. Server compares both answers, sends back who was right

**Key design insight — single source of truth for pixels:**
Both human and AI see *identical pixels* at *identical moments*, because both go through the same function `distortion.frame_at(image, mode, t, time_limit, seed)` with the same seed. The seed is a random integer chosen per-round on the server.

---

## 3. Dataset: Why Caltech-101?

### What is Caltech-101?

Caltech-101 is a classic computer vision benchmark dataset created by Fei-Fei Li, Marco Andreetto, and Marc'Aurelio Ranzato at Caltech in 2004. It contains:
- **9,144 images** across **101 object categories** (plus a background category)
- Images are real photographs, not illustrations
- Categories include animals, vehicles, flowers, household objects, musical instruments, and more

### Why it was chosen for this project

| Reason | Explanation |
|---|---|
| **Real photos** | The game needs authentic-looking images. Caltech-101 has clean, real photographs of single objects, perfect for a recognition game |
| **Diverse categories** | 85 usable classes across animals, vehicles, food/nature, objects — gives the game variety |
| **Right difficulty level** | Images are not trivially easy (like pure white-background product shots) but not impossibly hard either |
| **Unusual labels** | Classes like "Euphonium Horn", "Bonsai Tree", "Okapi" are not in ImageNet — this is the *exact reason* a plain CNN cannot be used as the AI (more on this below) |
| **Legally usable** | Caltech-101 is a research dataset with permissive use; no licensing issues |
| **Manageable size** | ~140 MB compressed — downloadable at game setup time |

### How the game uses it

The game maps Caltech-101 folder names to clean display names:
- `cougar_body` → "Cougar" (animal)
- `airplanes` → "Airplane" (vehicle)
- `bonsai` → "Bonsai Tree" (food/nature)
- `euphonium` → "Euphonium Horn" (object)

It then organises them into 4 super-categories: **animals, vehicles, food, objects**. Distractors (wrong options) are always drawn from the **same super-category**, making the game genuinely hard — you can't just pick "the only animal" in the list.

### Known weakness

Caltech-101 has a class imbalance problem: `airplanes` has 800 images and `Motorbikes` has 798 images — together that is ~23% of all rounds. The game notes this as a known gap.

---

## 4. The AI Model: CLIP ViT-B/32 — Explained

### What is CLIP?

**CLIP** (Contrastive Language–Image Pre-training) was created by OpenAI and published in 2021. It is a neural network trained on 400 million (image, text description) pairs scraped from the internet.

CLIP learns to put images and their text descriptions close together in a shared "embedding space" — a mathematical space where similar things have similar coordinates.

### What is ViT-B/32?

**ViT** = Vision Transformer. Instead of using convolutional filters (like ResNet), a ViT:
1. Splits the image into a grid of **fixed-size patches** (32×32 pixels for B/32)
2. Treats each patch like a word in a sentence
3. Runs a **Transformer** (the same architecture as GPT) over the sequence of patches
4. Produces a single vector (embedding) representing the whole image

**"B"** = Base size model (not Small, not Large). 87.8 million parameters in the visual encoder, 151.3 million total.

### Why CLIP instead of a plain ImageNet CNN?

This is the most important design decision and a very likely viva question:

| Problem | Plain CNN (e.g. ResNet-50) | CLIP ViT-B/32 |
|---|---|---|
| How does it classify? | Fixed list of 1000 ImageNet labels, chosen by argmax | Scores *any text string* directly against the image |
| Can it name "Bonsai Tree"? | ❌ No — "bonsai" is not an ImageNet class | ✅ Yes — it scores the text "a photo of a Bonsai Tree" |
| Can it name "Euphonium Horn"? | ❌ No | ✅ Yes |
| Needs label-to-class mapping? | ✅ Yes, and 37 of our 82 labels have no match | ❌ No mapping needed |
| Zero-shot capable? | ❌ No | ✅ Yes |

Only 45 of the game's 82 labels exist verbatim in ImageNet's 1000 classes. A plain CNN would be unable to play at all for the remaining 37 labels.

### How CLIP scores options in this game

For each of the 4 answer options (e.g. "Crocodile", "Sea Turtle", "Lobster", "Scorpion"):

1. CLIP encodes the text `"a photo of a crocodile."` → a vector
2. CLIP encodes the image → a vector
3. It computes the **dot product** (cosine similarity) between image vector and each text vector
4. The option with the highest similarity is CLIP's "first choice"
5. A **softmax** converts the 4 scores into probabilities that sum to 1.0

### Prompt ensemble

Instead of just `"a photo of a {label}."`, the game averages across 4 templates:
```
"a photo of a {}."
"a blurry photo of a {}."
"a low resolution photo of a {}."
"a cropped photo of a {}."
```

This is called **prompt ensembling** — a standard CLIP technique worth ~2 accuracy points. Since the images are corrupted, including "blurry" and "low resolution" templates is especially appropriate.

### Why `-quickgelu` matters

The code uses `MODEL_NAME = "ViT-B-32-quickgelu"`. This is important: OpenAI trained ViT-B/32 with a specific activation function called **QuickGELU**. If you load it with the plain `ViT-B-32` config, open_clip silently uses a different activation (GELU), which hurts accuracy. The `-quickgelu` config is the correct pairing for OpenAI weights.

---

## 5. Difficulty Calibration — The Three Honest Handicaps

### What "honest" means here

The difficulty is NOT a fake dial that secretly adjusts the AI's random number generator to lose 60% of the time. That would be dishonest game design. Instead, the AI is handicapped at the *mechanism* level — it genuinely sees worse inputs and has less information.

### The three parameters

#### `temperature` — how random the AI's final pick is

The AI's raw output is a probability distribution, e.g.:
```
Crocodile: 72%   Sea Turtle: 18%   Lobster: 6%   Scorpion: 4%
```

At **temperature=1.0**: AI picks from this distribution directly (picks "Crocodile" 72% of the time).

At **temperature=9.0** (Easy): The distribution is *flattened* — the 72% becomes closer to 25%, so the AI makes many more mistakes. But crucially, mistakes are *plausible* — it picks "Sea Turtle" far more often than "Scorpion", because the distribution shape is preserved, just compressed. A blurry crocodile never loses to a random unrelated option.

At **temperature=1.5** (Hard): Distribution is *sharpened* — the AI almost always picks its top prediction (plays close to argmax).

| Profile | Temperature | Effect |
|---|---|---|
| Easy | 9.0 | Very flat — lots of plausible mistakes |
| Medium | 2.5 | Slightly flat — competitive |
| Hard | 1.5 | Nearly argmax — almost never misses |

#### `sight_lag` — how many seconds behind the AI is

The human sees the frame at time `t`. The AI sees the frame from time `t - sight_lag`. So at Easy (1.0s lag), the AI always sees a *more corrupted* version of the image than the human.

At Hard (0.0s lag), the AI sees exactly what the human sees.

#### `lock_conf` — when the AI commits its answer

The AI looks at the image at 5 moments (called "glances") during the reveal. At each glance it checks: "Is my top confidence ≥ lock_conf?" If yes, it locks. If no, it keeps watching.

- Easy: `lock_conf = 0.60` — AI commits as soon as it has 60% confidence (often too early, with corrupted images)
- Hard: `lock_conf = 0.93` — AI waits until it's 93% sure (usually waits for a clearer frame)

The **decision time is emergent** — it's not hardcoded. It naturally depends on how clear the image is at each glance, which is why the mean lock time is different per corruption mode.

### Measured performance (from scripts/calibrate_ai.py, 40 rounds × 5 modes)

```
difficulty   pixelate   blur   noise   tiles   contrast   overall   mean lock
Easy              30%    48%     48%     30%        40%       39%      0.94s
Medium            62%    57%     82%     80%        70%       70%      0.96s
Hard              88%    80%     95%     95%        88%       89%      1.11s
```

**Key insight:** The AI varies 15–25 points across corruption modes *within each difficulty*. This is the model genuinely reacting to the images — not a programmed spread.

- **Pixelation and blur hurt ViT the most**: ViT splits the image into 32×32 patches. Pixelation destroys the within-patch structure (edges become block boundaries, not object edges). Blur similarly smears patch content. The Transformer can't find meaningful patterns in the patches.
- **TV noise barely hurts**: High-frequency random noise doesn't destroy the low-frequency shape information. The model can still recognise the silhouette even through noise.

---

## 6. The Five Corruption Modes — Code Explained Line by Line

### `distortion.py` — Overview

This file is the **single source of truth** for what every pixel looks like at every moment in a round. Both the human's pre-rendered frames and the AI's glances go through the same function.

```python
DISTORTION_MODES = [
    {"id": "pixelate", "name": "8-Bit Retro Mosaic", "icon": "👾"},
    {"id": "blur",     "name": "Foggy Defocus Blur",  "icon": "🌫️"},
    {"id": "noise",    "name": "TV Static Noise",      "icon": "📺"},
    {"id": "tiles",    "name": "Mystery Cutout Grid",  "icon": "🧩"},
    {"id": "contrast", "name": "Contrast Silhouette",  "icon": "🌓"}
]
```
This is a list of dictionaries. Each has an `id` (used internally), a display `name`, and an emoji `icon` for the UI.

---

### Mode 1: Blur (`apply_blur_mode`)

```python
def apply_blur_mode(image: Image.Image, progress: float) -> Image.Image:
```
- `image`: a PIL Image object (the photo)
- `progress`: a float from 0.0 (start, fully corrupted) to 1.0 (end, clear)

```python
    if progress >= 0.85:
        return image
```
At 85% of the round time, the image is returned completely clean. This gives the player a clear final moment.

```python
    intensity = (1.0 - (progress / 0.85)) ** 1.6
```
- `progress / 0.85` normalises progress to a 0–1 range *within* the corruption phase
- `1.0 - ...` flips it (1=fully corrupted at start, 0=clear at end)
- `** 1.6` applies a power curve — makes the clearing *accelerate* near the end (more dramatic reveal)

```python
    radius = max(0.0, 9.0 * intensity)
```
- Maximum blur radius is 9.0 pixels at the start, decreasing to 0 at the clear point
- `max(0.0, ...)` prevents negative radius

```python
    return image.filter(ImageFilter.GaussianBlur(radius=radius))
```
Pillow's Gaussian blur filter is applied. Gaussian blur averages each pixel with its neighbours, weighted by a bell curve. Larger radius = more blurring.

---

### Mode 2: Pixelate (`apply_pixelate_mode`)

```python
    orig_w, orig_h = image.size
```
Saves the original dimensions (448×448 in this game, because all images are pre-cropped to that size).

```python
    norm = min(1.0, progress / 0.85)
    factor = 0.08 + 0.92 * (norm ** 2.0)
```
- `norm`: progress clamped to [0, 1] within the corruption phase
- `factor`: scaling factor going from 0.08 (8% of full resolution = very blocky) to 1.0 (full resolution)
- The `** 2.0` (quadratic curve) means the early stages are very blocky and clearing accelerates

```python
    small = image.resize((small_w, small_h), resample=Image.Resampling.BILINEAR)
    return small.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
```
- Downscale to a small size using bilinear (smooth) resampling
- Then upscale back to original using **nearest-neighbour** (blocky) resampling
- This creates the classic "mosaic/8-bit pixel art" effect — each large block gets a solid colour

---

### Mode 3: Noise (`apply_noise_mode`)

```python
    arr = np.array(image.convert("RGB"), dtype=np.float32)
```
Converts the PIL image to a NumPy array of shape (height, width, 3) with float32 values for arithmetic.

```python
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 75 * intensity, arr.shape)
```
- Creates a **seeded** random number generator — same seed → same noise pattern every time
- `rng.normal(0, std, shape)` generates Gaussian noise (bell-curve distribution, centred at 0)
- Standard deviation is `75 * intensity` — at full corruption it's ±75 pixel values of noise (on a 0–255 scale)
- **Why seeded?** Because both the human and the AI must see identical pixels. Non-seeded noise would be different for each, making the contest unfair.

```python
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
```
- Adds noise to each pixel value
- `np.clip` prevents values going below 0 or above 255 (valid pixel range)

```python
    if intensity > 0.3:
        noisy_img = noisy_img.filter(ImageFilter.GaussianBlur(radius=1.2 * intensity))
```
A slight blur blends the grain — makes it look like film grain or TV static rather than pure random dots.

---

### Mode 4: Tiles (`apply_tiles_mode`)

```python
    grid_size = 5  # 5x5 = 25 tiles
```
Divides the image into a 5×5 grid of 25 tiles.

```python
    rng = random.Random(seed)
    all_tiles = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    rng.shuffle(all_tiles)
```
- Creates a seeded Python `random.Random`
- Makes a list of all 25 tile positions: `[(0,0), (0,1), ..., (4,4)]`
- Shuffles them with the seed — the *order* in which tiles are revealed is deterministic per round

```python
    norm = min(1.0, progress / 0.85)
    tiles_to_cover = int(len(all_tiles) * 0.75 * (1.0 - norm))
```
- At progress=0: cover `25 × 0.75 = 18` tiles (75% covered)
- At progress=0.85: cover 0 tiles (fully revealed)
- Tiles peel away linearly as time passes

```python
    draw.rectangle([x0, y0, x1, y1], fill=(15, 23, 42))
```
Draws dark slate-coloured rectangles over each covered tile. The colour `(15, 23, 42)` is a very dark blue-grey matching the game's UI theme.

---

### Mode 5: Contrast (`apply_contrast_mode`)

```python
    enhancer_con = ImageEnhance.Contrast(image)
    high_con = enhancer_con.enhance(1.0 + 2.2 * intensity)
```
- `Contrast(image).enhance(factor)` multiplies the contrast by `factor`
- At full intensity: contrast is 3.2× — extreme bright whites and deep blacks (silhouette effect)

```python
    enhancer_color = ImageEnhance.Color(high_con)
    result = enhancer_color.enhance(1.0 - 0.7 * intensity)
```
- `Color(image).enhance(factor)` adjusts saturation
- At full intensity: saturation is 0.3 (nearly greyscale)
- Combined with extreme contrast: image looks like a high-contrast black-and-white silhouette

```python
    if intensity > 0.2:
        result = result.filter(ImageFilter.GaussianBlur(radius=1.8 * intensity))
```
Adds a slight blur to soften harsh edges and make the silhouette look more dramatic.

---

### The master function: `frame_at`

```python
def frame_at(image, mode, t, time_limit, seed):
    progress = 0.0 if time_limit <= 0 else max(0.0, min(1.0, t / time_limit))
    return apply_distortion_by_mode(image, mode, progress=progress, seed=seed)
```
- `t / time_limit` converts an elapsed time into a 0–1 progress fraction
- `max(0.0, min(1.0, ...))` clamps it so it never goes below 0 or above 1
- This is the **only place** the reveal schedule is defined — neither the frontend nor the AI hardcode their own schedule

---

## 7. `ai_engine.py` — Line by Line

### Module-level constants

```python
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
```
Selects the OpenAI-pretrained weights with the correct activation function config (see §4).

```python
PROMPT_TEMPLATES = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a low resolution photo of a {}.",
    "a cropped photo of a {}.",
)
```
Four text templates for prompt ensembling. `{}` is replaced by the label name at runtime.

```python
GLANCE_FRACTIONS = (0.15, 0.32, 0.50, 0.68, 0.86)
```
The AI looks at the image at these 5 moments (as fractions of the round time). For a 3.5s round, that's 0.53s, 1.12s, 1.75s, 2.38s, 3.01s. All 5 frames are processed in a **single batched forward pass**.

---

### `DifficultyProfile` dataclass

```python
@dataclass(frozen=True)
class DifficultyProfile:
    key: str
    label: str
    temperature: float
    sight_lag: float
    lock_conf: float
```
A frozen dataclass — immutable after creation. Stores the three handicap parameters per difficulty level.

### The PROFILES dictionary

```python
PROFILES: dict[str, DifficultyProfile] = {
    "easy":   DifficultyProfile("easy",   "Easy",   temperature=9.0, sight_lag=1.00, lock_conf=0.60),
    "medium": DifficultyProfile("medium", "Medium", temperature=2.5, sight_lag=0.35, lock_conf=0.75),
    "hard":   DifficultyProfile("hard",   "Hard",   temperature=1.5, sight_lag=0.0,  lock_conf=0.93),
}
```
These values were **measured** (via `scripts/tune_profiles.py` and `scripts/calibrate_ai.py`), not guessed.

---

### `ClipOpponent.__init__`

```python
self.device = torch.device("cpu")
```
Forces CPU inference — represents an edge device with no GPU.

```python
self.model, _, self.preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained
)
```
Downloads/loads the CLIP model. Returns 3 things:
- The model itself
- A deprecated transform (ignored with `_`)
- `preprocess` — resize to 224×224, normalise, convert to tensor

```python
self.model.eval().to(self.device)
```
- `.eval()` sets inference mode (disables dropout)
- `.to(self.device)` moves weights to CPU

```python
self._text_cache: dict[str, torch.Tensor] = {}
```
Caches text embeddings. 82 labels computed once at startup and reused every round — saves ~4 seconds per round.

---

### `prime_labels` and `_embed_label`

```python
def prime_labels(self, labels: list[str]) -> None:
    for label in labels:
        self._embed_label(label)
```
Called once at server startup via the `lifespan` hook.

```python
def _embed_label(self, label: str) -> torch.Tensor:
    prompts = [t.format(label.lower()) for t in PROMPT_TEMPLATES]
    with torch.no_grad():
        feats = self.model.encode_text(self.tokenizer(prompts).to(self.device))
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalise each vector
        mean = feats.mean(dim=0)                          # average the 4 template vectors
        mean = mean / mean.norm()                         # L2 normalise the average
    self._text_cache[label] = mean
    return mean
```
- `torch.no_grad()` — no gradient tracking needed (inference only), saves memory
- L2 normalisation is required because CLIP uses cosine similarity (dot product of unit vectors)

---

### `evaluate_round` — the AI playing a round

```python
glance_times = [f * time_limit for f in GLANCE_FRACTIONS]
view_times = [max(0.0, t - profile.sight_lag) for t in glance_times]
```
- `glance_times`: when the AI looks (e.g. [0.53, 1.12, 1.75, 2.38, 3.01] seconds for 3.5s limit)
- `view_times`: shifts each time back by `sight_lag` — AI sees a more corrupted, older frame

```python
frames = [frame_at(image, distortion_mode, vt, time_limit, seed) for vt in view_times]
batch = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
```
- Generates 5 corrupted frames (PIL images)
- Applies CLIP's preprocessing to each, then stacks into a tensor of shape `[5, 3, 224, 224]`

```python
text = torch.stack([self._embed_label(o) for o in options]).to(self.device)
```
Gets cached text embedding for each of the 4 options. Shape: `[4, 512]`.

```python
with torch.no_grad():
    img_feats = self.model.encode_image(batch)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    logits = 100.0 * img_feats @ text.T  # [5 glances, 4 options]
```
- `encode_image` runs the ViT on all 5 frames → shape `[5, 512]`
- Matrix multiply image embeddings by text embeddings → shape `[5, 4]`
- `100.0` scaling matches CLIP's learned logit scale (makes softmax outputs meaningful)

```python
true_probs = logits.softmax(dim=-1)                       # honest belief
play_probs = (logits / profile.temperature).softmax(dim=-1)  # what it samples from
```
- `true_probs`: model's genuine confidence at each glance
- `play_probs`: temperature-modified distribution (flatter at Easy, sharper at Hard)

```python
lock_idx = len(glance_times) - 1
for i in range(len(glance_times)):
    if float(true_probs[i].max()) >= profile.lock_conf:
        lock_idx = i
        break
```
Finds the first glance where confidence crosses `lock_conf`. The AI commits at that glance (or defaults to the last one).

```python
pick = int(torch.multinomial(row_play, num_samples=1).item())
choice = options[pick]
```
- `torch.multinomial` samples one index from the probability distribution
- At high temperature (Easy), the AI often picks a non-top option

```python
severity = int(round(100 * (1.0 - min(1.0, view_t / (0.85 * time_limit)))))
```
How corrupted the image was when the AI locked. 100% = start (fully corrupted), 0% = clear.

---

## 8. `dataset.py` — Line by Line

```python
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "categories", "101_ObjectCategories")
```
Path to the Caltech-101 data relative to the script's location (`backend/data/categories/101_ObjectCategories/`).

```python
CLASS_MAPPINGS = {
    "Leopards": {"name": "Leopard", "category": "animals"},
    ...
}
```
Maps Caltech-101 folder names to clean display names and super-categories. Two folders may map to the same display name (e.g. `cougar_body` and `cougar_face` both → "Cougar").

```python
DATASET_INDEX = []
CATEGORY_POOLS = {"animals": set(), "vehicles": set(), "food": set(), "objects": set()}

if os.path.exists(DATA_DIR):
    for folder_name, meta in CLASS_MAPPINGS.items():
        ...
        for img_file in images:
            DATASET_INDEX.append({...})
```
Runs at **import time**. Walks all mapped folders, builds a flat list of all image metadata. `CATEGORY_POOLS` stores unique display names per category, used for generating distractors.

```python
RECENT_IMAGE_IDS = set()
```
Memory of recently shown images, to prevent repeats.

```python
def get_round_data(category: str = None) -> dict:
    candidates = [x for x in pool if x["id"] not in RECENT_IMAGE_IDS]
    if not candidates:
        RECENT_IMAGE_IDS.clear()
        candidates = pool
    chosen = random.choice(candidates)
```
Filters recent images. If pool exhausted, clears memory and starts fresh. Picks uniformly at random.

```python
    image = Image.open(chosen["file_path"]).convert("RGB")
    image = ImageOps.fit(image, (448, 448), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
```
- Opens and converts to RGB (some Caltech-101 images are greyscale)
- Centre-crops to 448×448 — same operation CLIP's own preprocessing does, keeping the duel fair

```python
    same_cat_pool = list(CATEGORY_POOLS[chosen["category"]] - {chosen["name"]})
    distractors = random.sample(same_cat_pool, 3)
    options = [chosen["name"]] + distractors
    random.shuffle(options)
```
Draws 3 distractors from the **same super-category**. Shuffles so the correct answer isn't always in the same position.

---

## 9. `server.py` — Line by Line

```python
@asynccontextmanager
async def lifespan(_app: FastAPI):
    ai_engine.prime_labels(all_labels())
    yield
```
FastAPI lifecycle hook — runs startup code before accepting requests. Pre-computes all 82 label embeddings (~4 seconds). `yield` hands control back to FastAPI.

```python
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", ...])
```
Allows the Vite dev server (port 5173) to call the API (port 8000). In production, both are on port 8000 so CORS doesn't apply.

```python
ACTIVE_ROUNDS: dict[str, dict] = {}
```
In-memory dictionary of active rounds, keyed by UUID. Known gap: no TTL, so abandoned rounds accumulate.

```python
@app.post("/api/round/next")
def next_round(req: NextRoundRequest):
    round_id = str(uuid.uuid4())
    seed = random.randint(1, 10_000_000)
```
Generates a unique round ID (UUID4) and a random seed for reproducible corruption.

```python
    ai_result = ai_engine.evaluate_round(
        image=image, options=options, ...
    )
```
The AI plays its round **synchronously on the server** before the response is sent. The client receives only the AI's `decision_time` — not the choice (which is revealed only on submit).

```python
    qualities = (68, 72, 78, 88)
    frames = [
        image_to_base64(frame_at(image, distortion_mode, f * time_limit, time_limit, seed), quality=q)
        for f, q in zip(FRAME_FRACTIONS, qualities)
    ]
```
Pre-renders 4 frames at increasing JPEG quality. Early frames are heavily corrupted (low quality unnoticeable), later frames get higher quality. Sending 4 pre-rendered frames avoids server-side rendering during gameplay.

```python
@app.post("/api/round/submit")
def submit_round(req: SubmitAnswerRequest):
    round_info = ACTIVE_ROUNDS.pop(req.round_id)
```
`pop` removes the round from memory — prevents submitting the same round twice.

```python
# Mount the built SPA last so it does not shadow the API routes.
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
```
Mounts the built React app as static files. **Must be last** — if mounted first, it would intercept `/api/...` routes.

---

## 10. `main.py` — Line by Line

```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
```
Adds `backend/` to Python's module search path so `server.py` can do `import ai_engine` (not `import backend.ai_engine`).

```python
PORT = int(os.environ.get("PORT", "8000"))
```
Reads port from environment variable, defaulting to 8000. Allows `PORT=8010 python main.py`.

```python
uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)
```
- `"server:app"` — find the `app` object inside the `server` module
- `host="0.0.0.0"` — listen on all network interfaces (accessible from LAN, not just localhost)
- `reload=False` — production mode; use `reload=True` for auto-restart during development

---

## 11. Scripts Explained

### `scripts/smoke_test.py`
Boots the app and plays several rounds through the HTTP API, asserting invariants like:
- The correct option is always in the 4 options
- Scores update correctly
- Submit fails for unknown round IDs

This is an **integration test** — tests the whole system end-to-end.

### `scripts/calibrate_ai.py`
Runs 40 rounds × 5 corruption modes × 3 difficulty levels automatically, records the AI's accuracy and mean lock time, and prints the performance table. This is how the difficulty profiles were **measured** rather than assumed.

### `scripts/tune_profiles.py`
Sweeps across different values of `temperature`, `sight_lag`, and `lock_conf` to find combinations that hit target accuracy ranges (Easy = ~40%, Medium = ~70%, Hard = ~90%). The `PROFILES` dictionary in `ai_engine.py` was filled from this sweep's output.

---

## 12. Viva Question Bank

### 🟢 Basic Questions (you must know these)

**Q: What does Vision Duel do in one sentence?**
A: It's a game where a human races a real AI to identify a corrupted image that progressively becomes clearer over 3.5 seconds.

**Q: Why can't you use a standard ImageNet CNN as the AI opponent?**
A: Because the game uses text labels as options, and only 45 of the 82 game labels exist in ImageNet's 1000 classes. A CNN with a fixed 1000-class output cannot name "Bonsai Tree" or "Euphonium Horn" at all. CLIP can score any text string directly against the image.

**Q: What is CLIP?**
A: Contrastive Language–Image Pre-training. A neural network trained on 400 million (image, text) pairs to place images and their descriptions close together in a shared mathematical space. It scores how well an image matches a text description.

**Q: What is Caltech-101?**
A: A benchmark dataset of 9,144 real photographs across 101 object categories, created at Caltech in 2004. Used here because it has diverse categories, authentic photos, and unusual labels that test CLIP's zero-shot ability.

**Q: What does "zero-shot" mean?**
A: The model was never shown Caltech-101 images during training. It classifies them at test time using only text descriptions, with no fine-tuning on these specific classes.

**Q: What are the 5 corruption modes?**
A: Blur (Gaussian), Pixelate (mosaic), Noise (TV static, Gaussian noise), Tiles (black tiles peeling away), Contrast (high-contrast silhouette).

**Q: Why is the noise seeded?**
A: Because both the human and the AI must see identical pixels at identical moments. If noise were random each time, one could see a clearer version than the other, making the contest unfair.

**Q: What are the three difficulty handicaps?**
A: `temperature` (how random the AI's pick is), `sight_lag` (how many seconds behind the AI sees), and `lock_conf` (how confident it needs to be before committing).

---

### 🟡 Intermediate Questions

**Q: What is a Vision Transformer (ViT)?**
A: A neural network that applies the Transformer architecture to images by splitting the image into fixed-size patches (32×32 pixels for B/32), treating each patch as a "token", and processing the sequence with self-attention layers.

**Q: Why does pixelation hurt ViT more than noise?**
A: ViT splits images into 32×32 patches and finds patterns within each patch. Pixelation destroys within-patch structure — edges become block boundaries. Noise is high-frequency and doesn't destroy the low-frequency shape information that lives within patches.

**Q: What is softmax temperature? How does it make Easy difficulty work?**
A: Temperature divides the logits before applying softmax. High temperature (9.0) flattens the probability distribution — all options become closer to equally likely. The AI samples from this flat distribution, frequently picking its 2nd or 3rd choice. Mistakes are still *plausible* — a blurry crocodile loses to "Sea Turtle", not to "Accordion".

**Q: What is prompt ensembling?**
A: Using multiple text templates (e.g. "a photo of a {}", "a blurry photo of a {}") and averaging their embeddings. This makes the text representation more robust by averaging out template-specific biases. Worth ~2 accuracy points.

**Q: What does `frame_at()` do and why is it important?**
A: It's the single function that maps (image, mode, time, time_limit, seed) → a corrupted PIL image. Both human and AI go through this same function with the same seed, guaranteeing identical pixels.

**Q: How does the AI's decision time emerge naturally?**
A: The AI examines the image at 5 preset moments. At each moment it checks if its top confidence exceeds `lock_conf`. The first moment it crosses the threshold is when it locks. So the decision time depends on the model's actual confidence, not a hardcoded value.

**Q: What is `sight_lag` and how does it handicap the AI?**
A: `sight_lag` (in seconds) shifts the AI's view into the past. If the human sees the frame at t=1.5s, the AI sees the frame from t=0.5s (1 second earlier, more corrupted). At Hard difficulty, sight_lag=0 so the AI sees exactly what the human sees.

**Q: Why are distractors from the same super-category?**
A: To make the game hard. If the answer is "Crocodile" and one option is "Airplane", you can eliminate it immediately. Same-category distractors (e.g. "Sea Turtle", "Crab", "Lobster") look visually similar and require genuine recognition.

**Q: What does `image_to_base64` do and why is it needed?**
A: It converts a PIL image to a base64-encoded JPEG string (a "data URI"). The browser receives JSON, not raw binary image files, so images must be text-encoded to travel in the JSON response body.

---

### 🔴 Advanced / Deep Questions

**Q: Why is `-quickgelu` important for the model config?**
A: OpenAI trained ViT-B/32 using the QuickGELU activation function. The plain `ViT-B-32` config in open_clip uses standard GELU instead. These produce different activation values, so using the wrong config silently degrades accuracy. The `-quickgelu` suffix selects the matching config for OpenAI weights.

**Q: Why is L2 normalisation applied to embeddings before computing similarity?**
A: Cosine similarity = dot(a, b) / (|a| × |b|). If both vectors are unit length (L2 norm = 1), cosine similarity = dot product. CLIP is trained with cosine similarity, so both image and text embeddings must be normalised. Without normalisation, the dot product would measure magnitude × similarity, not pure similarity.

**Q: The code computes `logits = 100.0 * img_feats @ text.T`. What is the 100.0 for?**
A: CLIP learns a temperature parameter during training, which settles around 100 (it's the reciprocal of the learned logit scale). Multiplying by 100.0 before softmax keeps the logit values in a range that produces meaningful probability distributions. Without it, logits would be tiny (cosine similarities are between -1 and 1), and softmax would output nearly uniform probabilities.

**Q: The AI plays its round synchronously before the HTTP response is sent. What are the implications?**
A: The HTTP request blocks until the AI finishes. Since the AI batch-processes all 5 glances in one forward pass (~5–50ms on CPU), this is acceptable. A non-blocking design would require storing partial state and a more complex API.

**Q: What are the known gaps/weaknesses of this system?**
A:
1. **No server-side deadline enforcement** — reaction times are client-supplied (easily cheatable)
2. **`ACTIVE_ROUNDS` has no TTL** — abandoned rounds accumulate forever (memory leak)
3. **Flat 1-point scoring** — no speed bonus, no difficulty multiplier
4. **Caltech-101 class imbalance** — Airplane (800 images) and Motorcycle (798 images) appear in ~23% of rounds
5. **Thin `food` and `vehicles` categories** — only 6 and 8 classes respectively
6. **No persistence** — scores reset on server restart

**Q: Why does the batch forward pass process all 5 glances at once rather than running the model 5 times?**
A: GPU/CPU inference has a fixed overhead per call (kernel launch, memory transfer). Running 5 images in a single batch is much faster — the overhead is paid once and compute is parallelised. The code notes: "the batch is an implementation detail, the agent only 'used' one glance to decide."

**Q: What research papers motivate this project?**
A:
- **Hendrycks & Dietterich, ICLR 2019** — "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations" — demonstrated CNNs fail catastrophically on corrupted images
- **Geirhos et al., NeurIPS 2018** — "ImageNet-trained CNNs are biased towards textures; increasing shape bias improves accuracy and robustness" — humans rely on shape, CNNs rely on texture

**Q: Why does the contrast mode use both contrast enhancement AND desaturation?**
A: High contrast alone creates vivid, harsh colours. The desaturation removes colour cues, so the player and model must rely purely on shape — the hardest possible recognition task. Together they create a stylised black-and-white silhouette.

**Q: What is `ImageOps.fit(image, (448, 448), method=LANCZOS, centering=(0.5, 0.5))` and why this size?**
A: It does a centre crop to exactly 448×448 pixels. This: (1) removes dead grey borders from Caltech-101's varied aspect ratios, (2) matches what CLIP's own preprocessing does (resize + centre crop), keeping the duel fair — human and AI see the same cropped region. 448 was chosen as a clean multiple of CLIP's patch size (32), larger than CLIP's 224 input (keeping more detail before downsampling).

---

### 🎯 Quick-fire Table (know these cold)

| Question | Answer |
|---|---|
| What port does the server run on? | 8000 (configurable via PORT env var) |
| How many corruption modes? | 5 |
| How many Caltech-101 classes used? | 85 mapped, 82 unique display labels |
| How many images total? | ~6,945 verified images |
| How many glances does the AI take? | 5 |
| How many frames are pre-rendered per round? | 4 |
| What is the default time limit? | 3.5 seconds |
| Where are CLIP weights cached? | `~/.cache/huggingface` |
| How big are the CLIP weights? | ~605 MB (open_clip_model.safetensors) |
| What does ACTIVE_ROUNDS store? | In-memory dict of ongoing rounds, keyed by UUID |
| What framework is the backend? | FastAPI + Uvicorn |
| What framework is the frontend? | React + Vite |
| What package manages Python deps? | uv |
| What does the lifespan hook do? | Pre-computes all 82 label text embeddings at startup |
| Easy AI accuracy overall? | ~39% |
| Medium AI accuracy overall? | ~70% |
| Hard AI accuracy overall? | ~89% |
| What is the `@dataclass(frozen=True)` for? | Makes DifficultyProfile immutable — parameters can't be changed after creation |
| What does `torch.no_grad()` do? | Disables gradient tracking during inference — saves memory and compute |
| What is `torch.multinomial`? | Samples an index from a probability distribution (not always the argmax) |

---

*Good luck in your viva! The three things to really nail:*
1. **Why CLIP over CNN** — it can score *text*, not just 1000 fixed labels
2. **How the three difficulty handicaps work mechanically** — temperature, sight_lag, lock_conf
3. **Why `frame_at()` is the single source of truth** — ensures human and AI see identical pixels
