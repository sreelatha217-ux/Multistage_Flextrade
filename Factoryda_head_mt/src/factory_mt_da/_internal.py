"""Internal package constants and logger."""

import logging

from .parameters import EPS
from .parameters import VERSION as __version__

log = logging.getLogger("factory_mt_da")

__all__ = ["EPS", "__version__", "log"]