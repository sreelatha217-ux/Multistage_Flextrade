"""Compatibility import and entry point for the modular BESS scheduler."""

from factory_mt_bess_da import *  # noqa: F403
from factory_mt_bess_da import __all__
from factory_mt_bess_da.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
