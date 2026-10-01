#!/usr/bin/env python3
"""Compatibility entry point for the :mod:`factory_da` package."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from factory_da import *  # noqa: F401,F403
from factory_da.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
