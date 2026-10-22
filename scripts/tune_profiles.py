"""
Sweep the difficulty parameters and report accuracy + lock time for each
combination, so ai_engine.PROFILES can be set from the curve.

The expensive part (CLIP) depends only on `sight_lag`, so logits are computed
once per (round, mode, sight_lag) and the temperature / lock_conf grid is then
evaluated analytically. That makes a full sweep seconds instead of minutes.

    uv run python scripts/tune_profiles.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import dataset as D  # noqa: E402
from ai_engine import GLANCE_FRACTIONS, ai_engine  # noqa: E402
from distortion import DISTORTION_MODES, frame_at  # noqa: E402

TIME_LIMIT = 3.5
ROUNDS = 24
SIGHT_LAGS = (0.0, 0.35, 0.65, 1.0)
TEMPERATURES = (0.8, 1.5, 2.5, 4.0, 6.0, 9.0)
LOCK_CONFS = (0.60, 0.75, 0.85, 0.93)


def logits_for(rd, mode, lag, seed):
    glance_times = [f * TIME_LIMIT for f in GLANCE_FRACTIONS]
    view_times = [max(0.0, t - lag) for t in glance_times]
    frames = [frame_at(rd["image"], mode, vt, TIME_LIMIT, seed) for vt in view_times]
    batch = torch.stack([ai_engine.preprocess(f) for f in frames])
    text = torch.stack([ai_engine._embed_label(o) for o in rd["options"]])
    with torch.no_grad():
        f = ai_engine.model.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
        return 100.0 * f @ text.T


def main() -> None:
    torch.manual_seed(0)
    ai_engine.prime_labels(D.all_labels())
    modes = [m["id"] for m in DISTORTION_MODES]
    rounds = [D.get_round_data("all") for _ in range(ROUNDS)]
    glance_times = [f * TIME_LIMIT for f in GLANCE_FRACTIONS]

    print(f"computing logits: {ROUNDS} rounds x {len(modes)} modes x {len(SIGHT_LAGS)} lags ...")
    cache = {}
    for lag in SIGHT_LAGS:
        for mode in modes:
            for i, rd in enumerate(rounds):
                cache[(lag, mode, i)] = (logits_for(rd, mode, lag, i), rd["options"], rd["correct_option"])

    print(f"\n{'lag':>5} {'temp':>6} {'conf':>6} | {'acc':>6} {'lock':>6}  | per-mode accuracy")
    print("-" * 86)
    for lag in SIGHT_LAGS:
        for temp in TEMPERATURES:
            for conf in LOCK_CONFS:
                per_mode, locks, hits = {}, [], []
                for mode in modes:
                    mode_hits = []
                    for i in range(ROUNDS):
                        logits, options, truth = cache[(lag, mode, i)]
                        true_p = logits.softmax(dim=-1)
                        idx = len(glance_times) - 1
                        for g in range(len(glance_times)):
                            if float(true_p[g].max()) >= conf:
                                idx = g
                                break
                        play = (logits[idx] / temp).softmax(dim=-1)
                        pick = int(torch.multinomial(play, 1).item())
                        mode_hits.append(options[pick] == truth)
                        locks.append(glance_times[idx])
                    per_mode[mode] = sum(mode_hits) / len(mode_hits)
                    hits.extend(mode_hits)
                acc = sum(hits) / len(hits)
                bar = " ".join(f"{m[:4]}:{per_mode[m] * 100:3.0f}" for m in modes)
                print(
                    f"{lag:>5.2f} {temp:>6.1f} {conf:>6.2f} | {acc * 100:>5.0f}% "
                    f"{sum(locks) / len(locks):>5.2f}s  | {bar}"
                )


if __name__ == "__main__":
    main()
