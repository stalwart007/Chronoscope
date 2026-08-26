#!/usr/bin/env python3
"""CLI wrapper around the packaged demo generator (``app.demo``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.demo import generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/uploads/sample_talk.mp4")
    args = parser.parse_args()
    out = generate(Path(args.out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {out.with_suffix('.srt')}")


if __name__ == "__main__":
    main()
