"""
End-to-end smoke test: boots the real app and plays a few rounds through the
HTTP API, asserting the invariants that matter.

    uv run python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures
    if not condition:
        failures += 1
    print(f"[{PASS if condition else FAIL}] {label}{f'  -- {detail}' if detail else ''}")


def main() -> None:
    with TestClient(server.app) as client:
        h = client.get("/api/health").json()
        check("health responds", h["status"] == "ok")
        check("model is a real CLIP opponent", "CLIP" in h["model"], h["model"])
        check("class count is distinct labels, not folders", h["stats"]["total_classes"] == 82,
              str(h["stats"]["total_classes"]))
        check("three difficulties exposed", len(h["difficulties"]) == 3)

        for time_limit in (2.5, 5.0):
            r = client.post("/api/round/next", json={
                "category": "all", "difficulty": "medium",
                "distortion_mode": "pixelate", "time_limit": time_limit,
            })
            check(f"round starts (limit {time_limit}s)", r.status_code == 200)
            d = r.json()

            check("4 options offered", len(d["options"]) == 4)
            check("4 reveal frames", len(d["frames"]) == 4)
            check("frame schedule is published to the client", len(d["frame_fractions"]) == 4)
            check("no answer leaked in the question",
                  not any(k in d for k in ("correct_option", "item_name", "ai_decision")))
            check("clear image not duplicated", "image_clear" not in d)
            check(f"AI locks inside the round ({d['ai_decision_time']}s <= {time_limit}s)",
                  d["ai_decision_time"] <= time_limit)

            s = client.post("/api/round/submit", json={
                "round_id": d["round_id"], "human_choice": d["options"][0], "reaction_time": 1.0,
            }).json()
            check("submit reveals ground truth", s["correct_option"] in d["options"])
            check("AI answered from the offered options", s["ai"]["choice"] in d["options"])

            probs = s["ai"]["probs"]
            check("real distribution over all 4 options", len(probs) == 4)
            check("probabilities sum to 1", abs(sum(probs.values()) - 1.0) < 0.01,
                  f"{sum(probs.values()):.4f}")
            check("reported confidence matches the chosen option's probability",
                  abs(probs[s["ai"]["choice"]] - s["ai"]["confidence"]) < 0.01)
            check("latency is measured, not invented", s["telemetry"]["inference_latency_ms"] > 0)

            again = client.post("/api/round/submit", json={"round_id": d["round_id"]})
            check("replaying a finished round is rejected", again.status_code == 404)

    print()
    if failures:
        print(f"{failures} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
