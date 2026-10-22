"""
Measure the AI opponent's real accuracy per difficulty profile and per
corruption mode, so the difficulty constants in ai_engine.PROFILES are set from
data rather than guessed.

This is also the project's core experiment: if corruption genuinely degrades the
model, accuracy must differ across modes. A flat row means the opponent is not
actually looking at the image.

    uv run python scripts/calibrate_ai.py [--rounds 30]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import dataset as D  # noqa: E402
from ai_engine import PROFILES, ai_engine  # noqa: E402
from distortion import DISTORTION_MODES  # noqa: E402

TIME_LIMIT = 3.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30, help="rounds per (difficulty, mode) cell")
    args = ap.parse_args()

    ai_engine.prime_labels(D.all_labels())
    modes = [m["id"] for m in DISTORTION_MODES]

    # Same rounds for every profile, so differences are the profile's doing.
    seeds = list(range(args.rounds))
    rounds = [D.get_round_data("all") for _ in seeds]

    print(f"\n{args.rounds} rounds per cell, {len(modes)} modes, {len(PROFILES)} profiles")
    print("=" * 78)
    header = f"{'profile':<12}" + "".join(f"{m:>11}" for m in modes) + f"{'overall':>11}"
    print(header)
    print("-" * 78)

    results: dict[str, dict[str, float]] = {}
    for key, profile in PROFILES.items():
        row, all_hits, lock_times = [], [], []
        for mode in modes:
            hits = []
            for seed, rd in zip(seeds, rounds):
                r = ai_engine.evaluate_round(
                    image=rd["image"],
                    options=rd["options"],
                    correct_option=rd["correct_option"],
                    distortion_mode=mode,
                    difficulty=key,
                    time_limit=TIME_LIMIT,
                    seed=seed,
                )
                hits.append(r["is_correct"])
                lock_times.append(r["decision_time"])
            acc = sum(hits) / len(hits)
            row.append(acc)
            all_hits.extend(hits)
        overall = sum(all_hits) / len(all_hits)
        results[key] = dict(zip(modes, row)) | {"overall": overall}
        line = f"{profile.label:<12}" + "".join(f"{a * 100:>10.0f}%" for a in row)
        print(f"{line}{overall * 100:>10.0f}%   lock~{statistics.mean(lock_times):.2f}s")

    print("-" * 78)
    spread = {
        k: max(v[m] for m in modes) - min(v[m] for m in modes) for k, v in results.items()
    }
    print("\nspread across corruption modes (0% would mean the image is ignored):")
    for k, s in spread.items():
        print(f"  {PROFILES[k].label:<12} {s * 100:.0f} points")

    print("\ntarget: ~65-70% for the default profile -- beatable but threatening.")


if __name__ == "__main__":
    main()
