"""
FastAPI server for Vision Duel.

One round = one image, one corruption mode, one seed, one clock. The human's
pre-rendered frames and the AI's glances are both produced by
`distortion.frame_at` from that single seed and clock, so both players are
looking at identical pixels at identical moments.
"""
import os
import random
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ai_engine import PROFILES, ai_engine
from dataset import all_labels, get_catalog_stats, get_round_data
from distortion import DISTORTION_MODES, frame_at, image_to_base64

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Encode every label once so rounds never pay the text-encoder cost."""
    ai_engine.prime_labels(all_labels())
    yield


app = FastAPI(title="Vision Duel", version="3.0.0", lifespan=lifespan)

# Same-origin in production (the SPA is mounted below) and proxied by Vite in
# development, so the only origin that needs listing is the dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fractions of the round at which the human's frames are rendered AND displayed.
# The frontend reads this off the round payload instead of hardcoding its own.
FRAME_FRACTIONS = (0.0, 0.30, 0.60, 0.90)

ACTIVE_ROUNDS: dict[str, dict] = {}


class NextRoundRequest(BaseModel):
    category: Optional[str] = "all"
    difficulty: Optional[str] = "easy"
    distortion_mode: Optional[str] = "auto"
    time_limit: float = Field(default=3.5, ge=1.0, le=15.0)


class SubmitAnswerRequest(BaseModel):
    round_id: str
    human_choice: Optional[str] = None
    reaction_time: Optional[float] = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": ai_engine.model_name,
        "dataset": "Caltech-101",
        "difficulties": [
            {"id": p.key, "label": p.label, "temperature": p.temperature}
            for p in PROFILES.values()
        ],
        "stats": get_catalog_stats(),
    }


@app.get("/api/categories")
def get_categories():
    return get_catalog_stats()


@app.post("/api/round/next")
def next_round(req: NextRoundRequest):
    round_id = str(uuid.uuid4())
    time_limit = req.time_limit
    seed = random.randint(1, 10_000_000)

    round_data = get_round_data(req.category)
    item = round_data["item"]
    image = round_data["image"]
    options = round_data["options"]
    correct_option = round_data["correct_option"]

    if not req.distortion_mode or req.distortion_mode == "auto":
        mode_obj = random.choice(DISTORTION_MODES)
    else:
        mode_obj = next(
            (m for m in DISTORTION_MODES if m["id"] == req.distortion_mode),
            DISTORTION_MODES[0],
        )
    distortion_mode = mode_obj["id"]

    # The AI plays the round against the same clock and seed as the human.
    ai_result = ai_engine.evaluate_round(
        image=image,
        options=options,
        correct_option=correct_option,
        distortion_mode=distortion_mode,
        difficulty=req.difficulty or "easy",
        time_limit=time_limit,
        seed=seed,
    )

    # The early frames are corrupted beyond the point where JPEG quality shows,
    # so only the final clear frame is worth spending bytes on.
    qualities = (68, 72, 78, 88)
    frames = [
        image_to_base64(
            frame_at(image, distortion_mode, f * time_limit, time_limit, seed),
            quality=q,
        )
        for f, q in zip(FRAME_FRACTIONS, qualities)
    ]

    ACTIVE_ROUNDS[round_id] = {
        "item_id": item["id"],
        "name": item["name"],
        "category": item["category"],
        "correct_option": correct_option,
        "options": options,
        "ai_result": ai_result,
    }

    return {
        "round_id": round_id,
        "category": item["category"],
        "options": options,
        "distortion": mode_obj,
        "frames": frames,
        "frame_fractions": list(FRAME_FRACTIONS),
        "time_limit": time_limit,
        "ai_decision_time": ai_result["decision_time"],
        "telemetry": ai_result["telemetry"],
    }


@app.post("/api/round/submit")
def submit_round(req: SubmitAnswerRequest):
    if req.round_id not in ACTIVE_ROUNDS:
        raise HTTPException(status_code=404, detail="Round expired or already submitted")

    round_info = ACTIVE_ROUNDS.pop(req.round_id)
    correct_option = round_info["correct_option"]
    ai_result = round_info["ai_result"]

    human_choice = req.human_choice
    human_correct = (human_choice == correct_option) if human_choice else False

    return {
        "correct_option": correct_option,
        "item_name": round_info["name"],
        "human": {
            "choice": human_choice,
            "is_correct": human_correct,
            "reaction_time": req.reaction_time,
            "points_earned": 1 if human_correct else 0,
        },
        "ai": {
            "choice": ai_result["choice"],
            "is_correct": ai_result["is_correct"],
            "decision_time": ai_result["decision_time"],
            "confidence": ai_result["confidence"],
            "reason": ai_result["reason"],
            "probs": ai_result["probs"],
            "points_earned": 1 if ai_result["is_correct"] else 0,
        },
        "telemetry": ai_result["telemetry"],
    }


# Mount the built SPA last so it does not shadow the API routes.
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
