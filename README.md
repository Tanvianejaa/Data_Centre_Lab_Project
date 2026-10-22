#  Vision Duel: AI vs. Human (DataCenter Edge Project)

An image recognition game where a human races a vision model to identify an object while the image
progressively de-corrupts over a countdown.

The premise is a real, documented phenomenon: **humans degrade gracefully under image corruption and
neural networks fall off a cliff** (Hendrycks & Dietterich, ICLR 2019; Geirhos et al., NeurIPS 2018).
The game is playable because of that gap, and it measures the gap while you play.

---

##  How a round works

1. The four options appear on their own in the middle of the screen, then slide into place.
   Reading them is not charged to your clock -- the photo and the countdown wait for you.
2. A Caltech-101 photo appears, heavily corrupted by one of five modes (pixelate, blur, TV noise,
   mystery tiles, contrast silhouette), and clarifies in four steps over a 3.5 second countdown.
3. Four options: the truth plus three distractors drawn from the same super-category.
4. You lock an answer with keys `1`–`4` or a click. The AI locks when it becomes confident enough.
5. Both answers are revealed. First to 3, 5 or 10 points wins.

The only things the player chooses are the target score and the difficulty. Everything else --
category, corruption mode, timing -- is handled for them. `Esc` or **Stop** halts a match; resuming
replays the current round so nobody gets to study a paused image.

---

##  The AI opponent

A **zero-shot CLIP ViT-B/32** classifier. It looks at the same corrupted frames you see, at the same
moments, and scores the four option strings directly.

**Why CLIP and not an ImageNet CNN:** the four options are *text*, so CLIP scores them with no
label-mapping step. Only 45 of the game's 82 labels exist verbatim in ImageNet's 1000 classes — a
plain classifier cannot name "Bonsai Tree" or "Euphonium Horn" at all.

### Difficulty is three honest handicaps, not a win-rate dial

| | What it does |
|---|---|
| `temperature` | The agent samples its answer from the model's real distribution at this temperature. High temperature produces mistakes, but *plausible* ones — a blurry crocodile loses to "Sea Turtle", never to a random option. |
| `sight_lag` | The agent is shown the frame from this many seconds earlier in the reveal than you see. |
| `lock_conf` | It watches the image clarify and commits at the first glance where its confidence crosses this bar, so decision time is **emergent**, not sampled from a hardcoded range. |

### Measured performance

`scripts/calibrate_ai.py`, 40 rounds × 5 corruption modes, 3.5s limit:

```
difficulty   pixelate   blur   noise   tiles   contrast   overall   mean lock
Easy              30%    48%     48%     30%        40%       39%      0.94s
Medium            62%    57%     82%     80%        70%       70%      0.96s
Hard              88%    80%     95%     95%        88%       89%      1.11s
```

Accuracy swings **15–25 points across corruption modes** inside every profile — that spread is the
model genuinely reacting to the image. Consistently, pixelation and blur hurt the ViT most (they
destroy within-patch structure) while high-frequency noise barely does.

---

##  Quick start

If `.venv/` already exists (it does on the original dev machine), you only need:

```bash
.venv/bin/python main.py
```

Then open **http://localhost:8000**. To use another port: `env PORT=8010 .venv/bin/python main.py`.

### From a fresh clone

```bash
uv venv && uv pip install -e .            # Python deps into .venv/
.venv/bin/python scripts/fetch_data.py    # Caltech-101 (~140 MB, not in git)
cd frontend && npm install && npm run build && cd ..
.venv/bin/python main.py
```

The first launch downloads the CLIP weights (~600 MB) into `~/.cache/huggingface`; later launches
start in about 5 seconds.

### Frontend development

```bash
cd frontend && npm run dev
```

Serves on port 5173 and proxies `/api` to 8000, so run the backend alongside it. The backend only
serves the React app from `frontend/dist`, so **after changing anything in `frontend/src` you must
run `npm run build`** for it to show up on port 8000.

---

##  Scripts

See **[docs/viva-guide.md](docs/viva-guide.md)** for the full explanation of the dataset choice,
the model, the difficulty calibration, the architecture and a viva question bank.

| Script | What it does |
|---|---|
| `scripts/smoke_test.py` | Boots the app and plays rounds through the HTTP API, asserting the invariants |
| `scripts/calibrate_ai.py` | Measures AI accuracy per difficulty × corruption mode |
| `scripts/tune_profiles.py` | Sweeps the difficulty parameters so profiles are set from the curve |

---

## 🛠️ Structure

```
backend/
  ai_engine.py    CLIP zero-shot opponent: glances, real probabilities, confidence-based locking
  dataset.py      Caltech-101 catalog, category mapping, distractor generation
  distortion.py   Five corruption modes + frame_at(), the single source of truth for the reveal clock
  server.py       FastAPI endpoints and static hosting
frontend/src/
  App.jsx         Game loop, RAF countdown, telemetry UI
  utils/sound.js  Web Audio synthesiser (no audio assets)
docs/
  design-brief.md Game design, dataset and model research
  code-audit.md   Audit of the first implementation and the remaining work
```

Both players' pixels come from one function: `distortion.frame_at(image, mode, t, time_limit, seed)`,
called with one seed per round. The frame schedule is published to the client in the round payload
(`frame_fractions`) rather than hardcoded in two places.

---

##  Status

Working: real CLIP opponent, measured difficulty calibration, shared reveal clock, five corruption
modes, pause/stop, smoke tests, and a UI rebuilt around the photograph.

Known gaps, tracked in [docs/code-audit.md](docs/code-audit.md): no server-side deadline enforcement
(reaction times are client-supplied), `ACTIVE_ROUNDS` has no TTL, flat 1-point scoring, Caltech-101
class imbalance (Airplane and Motorcycle are 23% of rounds), thin `food` and `vehicles` categories,
and no persistence of results.

The reveal response still carries the AI's full probability distribution (`ai.probs`) even though
the UI no longer shows it -- deliberately kept so a confidence panel can be added back for the
report without touching the backend.
