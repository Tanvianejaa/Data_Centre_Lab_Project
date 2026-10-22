import os
import sys

import uvicorn

# backend/ is not a package; put it on the path so its modules import cleanly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

PORT = int(os.environ.get("PORT", "8000"))


def main():
    print("\n" + "=" * 60)
    print("🚀 Vision Duel: AI vs Human (DataCenter Edge Project)")
    print(f"🎮 Open your browser at: http://localhost:{PORT}")
    print("🤖 Opponent: CLIP ViT-B/32 zero-shot (local CPU inference)")
    print("=" * 60 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    main()
