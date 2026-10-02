#!/usr/bin/env python3
"""Compatibility launcher for the modular day-ahead scheduler."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

main = importlib.import_module("factory_twostage.cli").main

if __name__ == "__main__":
    raise SystemExit(main())