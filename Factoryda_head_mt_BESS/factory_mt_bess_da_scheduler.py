"""Compatibility launcher for the modular factory MT+BESS scheduler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from factory_mt_bess_da import *  # noqa: F403
from factory_mt_bess_da import __all__
from factory_mt_bess_da.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
