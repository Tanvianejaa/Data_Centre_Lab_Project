"""
AI opponent for Vision Duel.

The opponent is a real zero-shot classifier. For each round it looks at the same
corrupted frames the human is looking at, at several moments during the reveal,
and produces a genuine probability distribution over the round's four options.

Why CLIP rather than a plain ImageNet CNN: the four options are *text*, so CLIP
scores them directly and no ImageNet-class -> game-label mapping is ever needed.
Only 45 of the 82 game labels exist verbatim in ImageNet's 1000 classes, so a
plain classifier simply cannot name "Bonsai Tree" or "Euphonium Horn".

Difficulty is NOT a win-rate constant. It is three honest handicaps:

  temperature   softmax temperature the agent samples its answer at. Low means
                it plays close to its true argmax; high means it makes mistakes,
                but *plausible* ones, because the distribution it samples from
                is the model's real one. A blurry crocodile loses to "Sea
                Turtle", never to a random option.
  sight_lag     seconds of extra corruption. The AI is shown the frame from
                `sight_lag` seconds earlier in the reveal than the human sees.
  lock_conf     the agent watches the image clarify and commits at the first
                glance where its confidence crosses this threshold, so decision
                time is emergent, not sampled from a hardcoded range.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import open_clip
from PIL import Image

from distortion import frame_at

# NB: the OpenAI weights were trained with QuickGELU. Pairing them with the
# plain "ViT-B-32" config silently swaps the activation and costs accuracy --
# open_clip warns about this. The "-quickgelu" config is the correct pairing.
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"

# Prompt ensemble. Averaging a few templates is standard practice for zero-shot
# CLIP and is worth a couple of accuracy points; it is computed once at startup.
PROMPT_TEMPLATES = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a low resolution photo of a {}.",
    "a cropped photo of a {}.",
)

# Moments during the reveal at which the agent looks at the image, as a fraction
# of the round length. Every glance is one batched forward pass, so the cost is
# a single inference call per round, not one per glance.
GLANCE_FRACTIONS = (0.15, 0.32, 0.50, 0.68, 0.86)


@dataclass(frozen=True)
class DifficultyProfile:
    """An opponent's handicaps. See the module docstring."""

    key: str
    label: str
    temperature: float
    sight_lag: float
    lock_conf: float


# Chosen off the sweep in scripts/tune_profiles.py, not guessed, then verified
# end to end by scripts/calibrate_ai.py over 40 rounds x 5 corruption modes at a
# 3.5s limit. Re-run that script after changing anything here.
#
#   profile    lag  temp  conf    measured acc   mean lock
#   Easy      1.00   9.0  0.60      35-39%       0.7-0.9s   half-blind and impulsive
#   Medium    0.35   2.5  0.75      70-73%       0.7-1.0s   confident early, a real contest
#   Hard      0.00   1.5  0.93      89-92%       0.9-1.1s   sees what you see, rarely misses
#
# Ranges are two independent 40-round runs. Each cell is 40 samples, so the
# standard error is around 8 points -- quote the range, not a single figure.
#
# Accuracy varies 15-25 points across corruption modes within every profile.
# That spread is the whole point: it is the model actually reacting to the
# image. The old coin-flip engine was flat at 35% across all five modes.
PROFILES: dict[str, DifficultyProfile] = {
    "easy": DifficultyProfile("easy", "Easy", temperature=9.0, sight_lag=1.00, lock_conf=0.60),
    "medium": DifficultyProfile("medium", "Medium", temperature=2.5, sight_lag=0.35, lock_conf=0.75),
    "hard": DifficultyProfile("hard", "Hard", temperature=1.5, sight_lag=0.0, lock_conf=0.93),
}
DEFAULT_PROFILE = "medium"


class ClipOpponent:
    """Zero-shot CLIP opponent that watches the reveal and commits when confident."""

    def __init__(self, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED):
        print(f"Loading AI opponent: CLIP {model_name}/{pretrained} ...")
        t0 = time.perf_counter()
        self.device = torch.device("cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model_name = f"CLIP {model_name.replace('-quickgelu', '')}"

        self.visual_params = sum(p.numel() for p in self.model.visual.parameters())
        self.total_params = sum(p.numel() for p in self.model.parameters())

        self._text_cache: dict[str, torch.Tensor] = {}
        print(
            f"AI opponent ready in {time.perf_counter() - t0:.1f}s "
            f"({self.visual_params / 1e6:.1f}M visual / {self.total_params / 1e6:.1f}M total params)"
        )

    # ------------------------------------------------------------------ text

    def prime_labels(self, labels: list[str]) -> None:
        """Pre-encode every label the game can ask about.

        Text embeddings never change, so encoding them once at startup turns the
        per-round text cost into a dictionary lookup.
        """
        t0 = time.perf_counter()
        for label in labels:
            self._embed_label(label)
        print(f"Cached {len(self._text_cache)} label embeddings in {time.perf_counter() - t0:.1f}s")

    def _embed_label(self, label: str) -> torch.Tensor:
        cached = self._text_cache.get(label)
        if cached is not None:
            return cached
        prompts = [t.format(label.lower()) for t in PROMPT_TEMPLATES]
        with torch.no_grad():
            feats = self.model.encode_text(self.tokenizer(prompts).to(self.device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0)
            mean = mean / mean.norm()
        self._text_cache[label] = mean
        return mean

    # ------------------------------------------------------------------ play

    def evaluate_round(
        self,
        image: Image.Image,
        options: list[str],
        correct_option: str,
        distortion_mode: str,
        difficulty: str = DEFAULT_PROFILE,
        time_limit: float = 3.5,
        seed: int = 0,
    ) -> dict:
        """Play one round and return the agent's genuine decision."""
        profile = PROFILES.get(difficulty, PROFILES[DEFAULT_PROFILE])

        glance_times = [f * time_limit for f in GLANCE_FRACTIONS]
        # The agent is handicapped by being shown an older, more corrupted frame.
        view_times = [max(0.0, t - profile.sight_lag) for t in glance_times]

        frames = [
            frame_at(image, distortion_mode, vt, time_limit, seed) for vt in view_times
        ]
        batch = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
        text = torch.stack([self._embed_label(o) for o in options]).to(self.device)

        start_ns = time.perf_counter_ns()
        with torch.no_grad():
            img_feats = self.model.encode_image(batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = 100.0 * img_feats @ text.T  # [glances, 4]
        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6

        true_probs = logits.softmax(dim=-1)  # the model's honest belief
        play_probs = (logits / profile.temperature).softmax(dim=-1)  # what it samples from

        # Watch the reveal; commit at the first glance that clears the bar.
        lock_idx = len(glance_times) - 1
        for i in range(len(glance_times)):
            if float(true_probs[i].max()) >= profile.lock_conf:
                lock_idx = i
                break

        # Real per-image latency, charged once -- the batch is an implementation
        # detail, the agent only "used" one glance to decide.
        per_image_ms = latency_ms / len(frames)
        decision_time = min(time_limit, glance_times[lock_idx] + per_image_ms / 1000.0)

        row_true = true_probs[lock_idx]
        row_play = play_probs[lock_idx]
        pick = int(torch.multinomial(row_play, num_samples=1).item())
        choice = options[pick]

        confidence = float(row_true[pick])
        top_idx = int(row_true.argmax())
        view_t = view_times[lock_idx]
        severity = int(round(100 * (1.0 - min(1.0, view_t / (0.85 * time_limit)))))

        if choice == correct_option:
            reason = f"Recognised it at {severity}% corruption."
        elif top_idx == pick:
            reason = f"Genuinely misread it as {choice} under {distortion_mode}."
        else:
            reason = f"Hesitated between {options[top_idx]} and {choice}; committed too early."

        return {
            "choice": choice,
            "is_correct": choice == correct_option,
            "decision_time": round(decision_time, 2),
            "confidence": round(confidence, 3),
            "reason": reason,
            # Real distribution over the four options, for the reveal panel.
            "probs": {opt: round(float(p), 4) for opt, p in zip(options, row_true)},
            "telemetry": {
                "model_name": self.model_name,
                "architecture": "Vision Transformer (zero-shot, text-matched)",
                "parameters": f"{self.visual_params / 1e6:.1f}M visual",
                "inference_latency_ms": round(per_image_ms, 2),
                "batch_latency_ms": round(latency_ms, 2),
                "glances": len(frames),
                "top_prediction_raw": options[top_idx],
                "distortion_mode": distortion_mode,
                "distortion_level_at_decision": f"{severity}%",
                "difficulty": profile.label,
                "temperature": profile.temperature,
                "sight_lag_s": profile.sight_lag,
                "hardware": "Edge Device CPU (Local)",
            },
        }


ai_engine = ClipOpponent()
