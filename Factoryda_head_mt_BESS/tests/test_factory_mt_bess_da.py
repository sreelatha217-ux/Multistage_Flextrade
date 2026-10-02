import json

import pytest

from factory_mt_bess_da import BESS, InstanceValidationError, RunConfig, build_candidates, build_model


def test_default_config_and_bess_benchmark():
    config = RunConfig.load("configs/default.json")
    instance = config.make_instance()

    assert RunConfig.bundled().to_dict() == config.to_dict()
    assert instance.bess is not None
    assert instance.bess.p_max_mw == 40.0
    assert instance.bess.e_max_mwh == 200.0
    assert instance.bess.round_trip == pytest.approx(0.76)
    assert RunConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_invalid_bess_initial_soc_is_rejected():
    with pytest.raises(InstanceValidationError, match="soc_init_mwh"):
        BESS(e_max_mwh=10.0).validate()

    with pytest.raises(InstanceValidationError, match="section 'solver'"):
        RunConfig.from_dict({"solver": []})


def test_benchmark_milp_contains_bess_state_and_mode_constraints():
    config = RunConfig.from_dict({"benchmark": {"n_tasks": 30}})
    instance = config.make_instance()
    candidates = build_candidates(instance, config.start_step_h)
    model = build_model(instance, config.scheduler_config(), candidates)

    assert len(model.Pch) == instance.horizon_h
    assert len(model.Pdis) == instance.horizon_h
    assert len(model.SoC) == instance.horizon_h
    assert len(model.c_soc_terminal) == 1
    assert len(model.c_bess_exclusive) == instance.horizon_h


def test_config_json_round_trip(tmp_path):
    config = RunConfig()
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    loaded = RunConfig.load(path)

    assert loaded.to_dict() == config.to_dict()
