"""Run one job: input.json -> model -> output.json. Always writes output.json, also on failure."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import mapper, output_builder, validate
from .errors import (InfeasibleSchedule, InvalidInput, RunTimeout, ServiceError, SolverFailure)

ROOT = Path(__file__).resolve().parent.parent          # folder that holds factory_mt_offering_strategy.py
MODEL_SCRIPT = ROOT / "factory_mt_offering_strategy.py"


def _status(workdir: Path, state: str, **kw) -> None:
    (workdir / "status.json").write_text(json.dumps({"state": state, "time": time.time(), **kw}))


def _last_error_line(stderr: str) -> str:
    lines = [ln for ln in stderr.splitlines() if " ERROR " in ln or "Error" in ln]
    return re.sub(r"^\S+\s+ERROR\s+", "", lines[-1]).strip() if lines else (stderr.strip().splitlines() or ["model failed"])[-1]


def run_model(argv: list[str], timeout_s: float) -> None:
    """Run the model as a separate process so a solver crash cannot take the service down."""
    try:
        proc = subprocess.run([sys.executable, str(MODEL_SCRIPT), *argv], cwd=ROOT, capture_output=True,
                              text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as err:
        raise RunTimeout(f"the model did not finish within {timeout_s:.0f} s") from err
    if proc.returncode == 0:
        return
    msg = _last_error_line(proc.stderr)
    if proc.returncode == 2:
        raise InvalidInput(msg)
    if proc.returncode == 3:
        raise (InfeasibleSchedule if "Infeasible" in msg else SolverFailure)(msg)
    raise SolverFailure(f"model stopped with exit code {proc.returncode}: {msg}")


def run(input_path: str | Path, output_path: str | Path, workdir: str | Path | None = None) -> int:
    """Returns 0 on success and 1 on failure; output.json is written in both cases."""
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="offering_job_"))
    wd.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        _status(wd, "running", step="reading input")
        try:
            d = json.loads(Path(input_path).read_text())
        except (OSError, json.JSONDecodeError) as err:
            raise InvalidInput(f"cannot read input file: {err}", "input") from err
        warnings = validate.validate_input(d)
        _status(wd, "running", step="preparing data")
        argv, info = mapper.build_files(d, wd)
        sv = (d.get("run_settings") or {}).get("solver") or {}
        limit = float(sv.get("time_limit_s", 300))
        timeout = float(sv.get("overall_timeout_s", limit * 8 + 120))   # the model runs several solves
        _status(wd, "running", step="solving")
        run_model(argv, timeout)
        _status(wd, "running", step="building output")
        out = output_builder.build_output(
            info["out_dir"], bool((d.get("run_settings") or {}).get("include_scenario_detail", False)),
            float(sv.get("mip_gap", 0.01)), warnings + info["warnings"], info.get("database"))
        out["meta"]["total_time_s"] = round(time.time() - started, 2)
        Path(output_path).write_text(json.dumps(out, indent=2))
        _status(wd, "done")
        return 0
    except ServiceError as err:
        body = {"meta": {"status": "error", "error": err.to_dict(), "warnings": []}}
    except Exception as err:                           # never leave the UI without an output file
        body = {"meta": {"status": "error", "error": {"code": "INTERNAL", "message": f"{type(err).__name__}: {err}"},
                         "warnings": []}}
    Path(output_path).write_text(json.dumps(body, indent=2))
    _status(wd, "error", error=body["meta"]["error"])
    return 1
