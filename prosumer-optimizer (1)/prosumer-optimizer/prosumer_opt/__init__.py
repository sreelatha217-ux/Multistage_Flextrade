"""
prosumer_opt - multi-stage stochastic MILP for an industrial prosumer
(batch production + BESS + microturbine) trading on the Day-Ahead, Intraday and
Real-Time balancing markets. All money in EUR.

Typical use::

    from prosumer_opt import AppConfig, run_pipeline
    result = run_pipeline(AppConfig())
    print(result.day_ahead.summary())

or, with full control over the inputs::

    from prosumer_opt import MultiStageProsumerOptimizer
"""
from ._version import __version__
from .config import (AppConfig, DemoFactorySettings, IntradaySettings, OutputSettings,
                     ScenarioSettings)
from .exceptions import (ConfigurationError, DataValidationError, ModelInfeasibleError,
                         ProsumerOptimizationError, SolveFailedError, SolverUnavailableError)
from .optimizer import MultiStageProsumerOptimizer
from .parameters import (BatchJob, BESSParams, FactoryParams, GridParams, MicroturbineParams,
                         SolverSettings, TimeGrid)
from .pipeline import PipelineResult, run_pipeline
from .results import StageResult
from .scenarios import IDScenario, RTBranch, ScenarioTree
from .state import DAPlan, InitialState, IntradayInputs

__all__ = [
    "__version__", "AppConfig", "DemoFactorySettings", "IntradaySettings", "OutputSettings",
    "ScenarioSettings", "MultiStageProsumerOptimizer", "run_pipeline", "PipelineResult",
    "TimeGrid", "GridParams", "MicroturbineParams", "BESSParams", "BatchJob", "FactoryParams",
    "SolverSettings", "RTBranch", "IDScenario", "ScenarioTree", "InitialState", "IntradayInputs",
    "DAPlan", "StageResult", "ProsumerOptimizationError", "ConfigurationError",
    "DataValidationError", "SolverUnavailableError", "ModelInfeasibleError", "SolveFailedError",
]
