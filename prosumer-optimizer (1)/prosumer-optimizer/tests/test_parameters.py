import pytest

from prosumer_opt import BatchJob, BESSParams, FactoryParams, MicroturbineParams, TimeGrid
from prosumer_opt.exceptions import ConfigurationError, DataValidationError


class TestTimeGrid:
    def test_steps_and_rounding_up(self):
        tg = TimeGrid(24, 0.5)
        assert tg.n_steps == 48
        assert tg.steps(1.5) == 3
        assert tg.steps(1.6) == 4          # durations round UP (conservative)
        assert tg.hour(5) == 2.5

    def test_horizon_must_be_multiple_of_dt(self):
        with pytest.raises(ConfigurationError, match="integer multiple"):
            TimeGrid(24, 0.7)


class TestMicroturbine:
    def test_effective_segments_clipped_above_pmin(self):
        segs = MicroturbineParams().effective_segments()
        assert [round(w, 6) for w, _ in segs] == [2.5, 7.2, 5.3]   # first block sits below P_min = 10 MW
        assert sum(w for w, _ in segs) == pytest.approx(25.0 - 10.0)

    def test_no_load_cost_is_fuel_cost_of_pmin(self):
        expected = 5.30 * 48.41 + 4.70 * 48.78
        assert MicroturbineParams().no_load_cost_eur_per_h() == pytest.approx(expected)

    def test_rejects_non_convex_cost_curve(self):
        with pytest.raises(ConfigurationError, match="non-decreasing"):
            MicroturbineParams(cost_blocks=((10.0, 50.0), (15.0, 40.0)))

    def test_rejects_blocks_not_covering_pmax(self):
        with pytest.raises(ConfigurationError, match="do not cover"):
            MicroturbineParams(cost_blocks=((5.0, 50.0),))

    def test_rejects_pmin_above_pmax(self):
        with pytest.raises(ConfigurationError):
            MicroturbineParams(p_min_mw=30.0)


class TestBESS:
    def test_profiles_match_reference_database(self):
        big, small = BESSParams.large_scale(), BESSParams.medium_scale()
        assert (big.p_max_mw, big.e_max_mwh, big.soc_min_mwh, big.soc_init_mwh) == (40.0, 200.0, 20.0, 20.0)
        assert (small.p_max_mw, small.e_max_mwh, small.soc_max_mwh, small.soc_init_mwh) == (2.0, 4.0, 3.6, 0.63)
        assert big.eta_ch * big.eta_dis == pytest.approx(0.76)      # round-trip efficiency

    def test_terminal_target_defaults_to_initial_soc(self):
        assert BESSParams().terminal_target == BESSParams().soc_init_mwh
        assert BESSParams(terminal_soc_mwh=50.0).terminal_target == 50.0

    def test_unknown_profile(self):
        with pytest.raises(ConfigurationError, match="unknown profile"):
            BESSParams.from_profile("huge")

    def test_rejects_initial_soc_outside_window(self):
        with pytest.raises(ConfigurationError, match="soc_init"):
            BESSParams(soc_init_mwh=5.0)          # below soc_min = 20


class TestFactory:
    def test_duplicate_ids_rejected(self):
        j = BatchJob("A", "M1", 0, 2.0, 50.0)
        with pytest.raises(ConfigurationError, match="duplicate job_id"):
            FactoryParams(jobs=[j, BatchJob("A", "M2", 0, 2.0, 50.0)])
        with pytest.raises(ConfigurationError, match="duplicate .machine, sequence"):
            FactoryParams(jobs=[j, BatchJob("B", "M1", 0, 2.0, 50.0)])

    def test_series_helpers(self):
        f = FactoryParams(base_load_mw=3.0, demand_units=[0, 1, 2])
        assert list(f.base_series(3)) == [3.0, 3.0, 3.0]
        assert list(f.demand_series(3)) == [0, 1, 2]
        with pytest.raises(DataValidationError):
            f.demand_series(4)

    def test_negative_load_rejected(self):
        with pytest.raises(DataValidationError, match=">= 0"):
            FactoryParams(base_load_mw=-1.0).base_series(4)
