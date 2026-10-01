"""Input data: reference tariff and synthetic demo generators."""
from .synthetic import build_demo_factory, build_demo_intraday_scenario, build_demo_tree
from .tariffs import tou_prices

__all__ = ["tou_prices", "build_demo_factory", "build_demo_tree", "build_demo_intraday_scenario"]
