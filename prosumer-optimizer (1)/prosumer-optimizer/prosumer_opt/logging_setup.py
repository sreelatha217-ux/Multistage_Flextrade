"""
Logging configuration for the command line interface.

Only the ``prosumer_opt`` logger is configured. Handlers that belong to the host process
(root logger, pytest, Jupyter, an embedding application) are never removed or closed:
tearing them down mid-run can break unrelated code, and with solver output captured by
Pyomo it can even deadlock.
"""
from __future__ import annotations

import logging
import sys

LOGGER_NAME = "prosumer_opt"
_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is *at emit time*, so a replaced or closed stream
    (test capture, redirected output) can never leave a stale handle behind."""

    def __init__(self):
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stderr


def configure_logging(verbose: bool = False, level: str = "INFO") -> None:
    """Idempotent: calling it again replaces the handler it installed earlier, nothing else."""
    lvl = logging.DEBUG if verbose else getattr(logging, str(level).upper(), logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)
    for h in [h for h in logger.handlers if isinstance(h, _StderrHandler)]:
        logger.removeHandler(h)
    handler = _StderrHandler()
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(lvl)
