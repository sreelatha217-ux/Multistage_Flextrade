"""Compatibility launcher for the packaged day-ahead scheduler."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from factory_multistage.day_ahead import *  # noqa: F403
from factory_multistage.day_ahead import _arr
from factory_multistage.cli import main


if __name__ == "__main__":
    raise SystemExit(main())