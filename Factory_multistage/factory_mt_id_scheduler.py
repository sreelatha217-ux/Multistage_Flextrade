"""Compatibility launcher for the packaged intraday scheduler."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from factory_multistage.intraday import *  # noqa: F403
from factory_multistage.intraday import _parse, _val
from factory_multistage.intraday_cli import main


if __name__ == "__main__":
    raise SystemExit(main())