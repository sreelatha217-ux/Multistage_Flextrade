import json

import pytest

from prosumer_opt.cli import apply_overrides, build_parser, main
from prosumer_opt.config import AppConfig


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_dump_config_prints_valid_json(capsys):
    assert main(["--dump-config"]) == 0
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["solver"]["name"] == "appsi_highs" and cfg["time"]["dt_h"] == 1.0


def test_flags_override_file_which_overrides_defaults(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("solver: {mip_gap: 0.01, time_limit_s: 50}\nscenarios: {n_id_scenarios: 7}\n")
    cfg = AppConfig.from_file(f)
    assert cfg.solver.mip_gap == 0.01 and cfg.scenarios.n_id_scenarios == 7      # file beats default
    cfg = apply_overrides(cfg, parse("--mip-gap", "0.001", "--rt-branches", "2"))
    assert cfg.solver.mip_gap == 0.001                                            # flag beats file
    assert cfg.solver.time_limit_s == 50 and cfg.scenarios.n_id_scenarios == 7    # untouched values survive
    assert cfg.scenarios.n_rt_branches == 2


def test_seed_flag_derives_all_three_seeds():
    cfg = apply_overrides(AppConfig(), parse("--seed", "100"))
    assert (cfg.demo_factory.seed, cfg.scenarios.seed, cfg.intraday.seed) == (100, 104, 192)


def test_shift_window_flag_reaches_demo_factory():
    from prosumer_opt.pipeline import resolve_factory
    cfg = apply_overrides(AppConfig(), parse("--shift-window", "0"))
    assert resolve_factory(cfg).intraday_shift_window_h == 0


def test_exit_code_2_for_bad_configuration(tmp_path):
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("solvr: {}\n")
    assert main(["--config", str(bad)]) == 2


def test_exit_code_3_for_unavailable_solver():
    assert main(["--solver", "no_such_solver", "--intraday-step", "0", "--no-save"]) == 3


def test_end_to_end_run_writes_all_outputs(tmp_path, capsys):
    out = tmp_path / "res"
    rc = main(["--id-scenarios", "2", "--rt-branches", "2", "--intraday-step", "10",
               "--mip-gap", "1e-3", "--out", str(out)])
    assert rc == 0
    names = {p.name for p in out.iterdir()}
    for stage in ("da", "id"):
        for suffix in ("schedule.csv", "batch_plan.csv", "scenario_detail.csv", "scenario_costs.csv", "summary.json"):
            assert f"{stage}_{suffix}" in names
    assert "config_used.json" in names
    assert AppConfig.from_dict(json.loads((out / "config_used.json").read_text())).scenarios.n_id_scenarios == 2
    assert "[DA] status=optimal" in capsys.readouterr().out
