"""Entry point for the scheduled live paper-trading tick.

Usage: python scripts/live_tick.py
Exit code 0 even on fetch failures (the next tick catches up); non-zero
only for unexpected errors so the scheduler can flag genuine breakage.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.live.decision_loop import run_tick


def main() -> int:
    summary = run_tick()
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
