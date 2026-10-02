"""Public three-stage day-ahead, intraday, and real-time scheduling API."""

from . import intraday as _intraday
from .intraday import *  # noqa: F403
from .realtime_cli import _parse, main
from .realtime_model import build_realtime_model
from .realtime_optimizer import optimize_realtime, value_of_rt_modelling
from .realtime_plotting import plot_realtime
from .realtime_results import RealTimeResult, extract_realtime
from .realtime_scenarios import MODES, BalancingMarket, RealTimeSet
from .realtime_testing import selftest

__all__ = [
    *_intraday.__all__,
    "MODES",
    "BalancingMarket",
    "RealTimeResult",
    "RealTimeSet",
    "build_realtime_model",
    "extract_realtime",
    "main",
    "optimize_realtime",
    "plot_realtime",
    "selftest",
    "value_of_rt_modelling",
    "_parse",
]