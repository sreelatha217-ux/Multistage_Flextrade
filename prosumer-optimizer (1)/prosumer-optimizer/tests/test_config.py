import json
from pathlib import Path

import pytest

from prosumer_opt import AppConfig, BESSParams
from prosumer_opt.exceptions import ConfigurationError

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_default_yaml_equals_builtin_defaults():
    assert AppConfig.from_file(CONFIGS / "default.yaml") == AppConfig()


def test_roundtrip_through_dict_and_json():
    cfg = AppConfig()
    assert AppConfig.from_dict(cfg.to_dict()) == cfg
    assert AppConfig.from_dict(json.loads(cfg.to_json())) == cfg


def test_partial_override_keeps_other_defaults():
    cfg = AppConfig.from_dict({"solver": {"mip_gap": 0.01}, "bess": {"profile": "medium"}})
    assert cfg.solver.mip_gap == 0.01 and cfg.solver.name == "appsi_highs"
    assert cfg.bess == BESSParams.medium_scale()


def test_bess_profile_with_field_override():
    cfg = AppConfig.from_dict({"bess": {"profile": "medium", "throughput_cost_eur_mwh": 45.0}})
    assert cfg.bess.p_max_mw == 2.0 and cfg.bess.throughput_cost_eur_mwh == 45.0


def test_medium_and_custom_factory_example_files_load():
    assert AppConfig.from_file(CONFIGS / "medium_bess.yaml").time.dt_h == 0.5
    cfg = AppConfig.from_file(CONFIGS / "custom_factory.yaml")
    assert [j.job_id for j in cfg.factory.jobs] == ["F1-B1", "F1-B2", "F2-B1"]
    assert len(cfg.factory.base_load_mw) == 24


@pytest.mark.parametrize("data, msg", [
    ({"solvr": {}}, "unknown top-level"),
    ({"solver": {"mipgap": 1e-3}}, "unknown key"),
    ({"bess": {"profile": "huge"}}, "unknown profile"),
    ({"bess": {"colour": "red"}}, "unknown key"),
    ({"time": {"dt_h": 0.7}}, "integer multiple"),
    ({"solver": {"mip_gap": 2.0}}, "mip_gap"),
    ({"microturbine": {"cost_blocks": [[10, 50], [10, 40]]}}, "non-decreasing"),
    ({"factory": {"jobs": [{"job_id": "A"}]}}, "factory.jobs"),
    ({"factory": "nope"}, "expected a mapping"),
])
def test_invalid_configs_fail_loudly(data, msg):
    with pytest.raises(ConfigurationError, match=msg):
        AppConfig.from_dict(data)


def test_file_errors(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        AppConfig.from_file(tmp_path / "missing.yaml")
    bad = tmp_path / "c.txt"
    bad.write_text("x")
    with pytest.raises(ConfigurationError, match="unsupported"):
        AppConfig.from_file(bad)
    broken = tmp_path / "c.json"
    broken.write_text("{not json")
    with pytest.raises(ConfigurationError, match="cannot parse"):
        AppConfig.from_file(broken)
    lst = tmp_path / "l.yaml"
    lst.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigurationError, match="mapping"):
        AppConfig.from_file(lst)
