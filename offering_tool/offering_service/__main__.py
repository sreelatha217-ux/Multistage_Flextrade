"""Command line:
    python -m offering_service run  --input input.json --output output.json [--workdir DIR]
    python -m offering_service test-db        # connection check for the 'Test connection' button
"""
from __future__ import annotations

import argparse
import sys

from . import db_market, runner
from .errors import ServiceError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="offering_service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one job")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--workdir")
    sub.add_parser("test-db", help="check the PostgreSQL connection")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return runner.run(a.input, a.output, a.workdir)
    try:
        print(db_market.test_connection())
        return 0
    except ServiceError as err:
        print(f"{err.code}: {err.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
