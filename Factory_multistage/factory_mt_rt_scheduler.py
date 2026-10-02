"""Compatibility launcher for the packaged real-time scheduler."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from factory_multistage.realtime import *  # noqa: F403
from factory_multistage.realtime import _parse
from factory_multistage.realtime_cli import main


if __name__ == "__main__":
    raise SystemExit(main())