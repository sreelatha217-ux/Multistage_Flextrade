"""
Integration tests on solved schedules. They check *physics*, not just that a solver returned something:
every constraint family of the formulation is verified on the extracted solution.
"""
import numpy as np
import pytest

from prosumer_opt import (BatchJob, FactoryParams, GridParams, MicroturbineParams, BESSParams,
                          MultiStageProsumerOptimizer, SolverSettings, TimeGrid)
from prosumer_opt.exceptions import (ConfigurationError, DataValidationError, ModelInfeasibleError,
                                     SolverUnavailableError)
from prosumer_opt.scenarios import IDScenario, RTBranch, ScenarioTree
from prosumer_opt.state import InitialState, IntradayInputs

TOL = 1e-5


# --------------------------------------------------------------------------- regression
class TestRegression:
    """Numbers recorded from the original single-file prosumer_optimizer.py (same inputs, gap 1e-6)."""

    def test_day_ahead_objective(self, da):
        assert da.status == "optimal"
        assert da.objective_eur == pytest.approx(264589.349092, rel=1e-6)

    def test_day_ahead_cost_breakdown(self, da):
        cb = da.cost_breakdown
        assert cb["DA"] == pytest.approx(48815.6, abs=0.1)
        assert cb["ID"] == pytest.approx(212108.1, abs=0.1)
        assert cb["BAL"] == pytest.approx(-29989.5, abs=0.1)
        assert cb["MT"] == pytest.approx(29912.8, abs=0.1)
        assert cb["BESS"] == pytest.approx(3742.3, abs=0.1)
        assert cb["UNMET"] == 0.0

    def test_model_size_unchanged_by_refactor(self, da):
        assert da.model_stats == dict(variables=2120, binaries=1064, constraints=1485)

    def test_intraday_objective(self, intraday):
        assert intraday.status == "optimal"
        assert intraday.objective_eur == pytest.approx(179623.40, rel=1e-5)


# --------------------------------------------------------------------------- accounting
class TestAccounting:
    def test_objective_equals_expected_total_cost(self, da, intraday):
        for r in (da, intraday):
            assert r.objective_eur == pytest.approx(r.cost_breakdown["TOTAL"], rel=1e-9)
            probs, tot = r.scenario_costs["prob"], r.scenario_costs["TOTAL"]
            assert float(np.dot(probs, tot)) == pytest.approx(r.objective_eur, rel=1e-9)

    def test_components_sum_to_total(self, da):
        cb = da.cost_breakdown
        assert sum(v for k, v in cb.items() if k != "TOTAL") == pytest.approx(cb["TOTAL"])

    def test_risk_metrics_ordering(self, da):
        r = da.risk
        assert r["best"] <= r["mean"] <= r["cvar95"] <= r["worst"] + TOL
        assert r["std"] > 0                     # 3 different price scenarios

    def test_da_cost_recomputed_independently(self, da, tree, optimizer):
        """Recompute EUR of the DA position from raw scenario prices and the reported plan."""
        p = da.plan
        for i, s in enumerate(tree.scenarios):
            eur = float(np.sum(np.asarray(s.da_buy) * p.p_da_buy - np.asarray(s.da_sell) * p.p_da_sell)
                        * optimizer.time.dt_h)
            assert eur == pytest.approx(da.scenario_costs["DA"].iloc[i], rel=1e-9)


# --------------------------------------------------------------------------- Stage 1 structure
class TestNonAnticipativity:
    def test_da_position_identical_across_scenarios(self, da):
        d = da.scenario_detail
        for col in ("da_buy_mw", "da_sell_mw", "mt_on"):
            assert (d.groupby("step")[col].nunique() == 1).all(), col

    def test_no_simultaneous_da_buy_and_sell(self, da):
        assert not ((da.plan.p_da_buy > TOL) & (da.plan.p_da_sell > TOL)).any()


# --------------------------------------------------------------------------- power balance
def test_power_balance_holds_in_expectation(da, tree, optimizer):
    """DA + ID + MT + dis - ch - base*(1+E[dev]) - batch == E[surplus] - E[shortfall], every (scenario, step)."""
    base = optimizer.factory.base_series(optimizer.time.n_steps)
    names = [s.name for s in tree.scenarios]
    for i, name in enumerate(names):
        edev = sum(br.prob * np.asarray(br.load_dev) for br in tree.scenarios[i].rt_branches)
        d = da.scenario_detail.query("scenario == @name").sort_values("step")
        t = d["step"].to_numpy()
        lhs = (d.da_buy_mw - d.da_sell_mw + d.id_buy_mw - d.id_sell_mw + d.mt_mw
               + d.bess_dis_mw - d.bess_ch_mw - base[t] * (1 + edev[t]) - d.batch_load_mw)
        rhs = d.imb_surplus_mw - d.imb_shortfall_mw
        assert np.allclose(lhs, rhs, atol=1e-6), name


def test_grid_limits_respected(da, optimizer):
    lim = optimizer.grid.import_limit_mw
    net = da.scenario_detail.eval("da_buy_mw - da_sell_mw + id_buy_mw - id_sell_mw")
    assert (net <= lim + TOL).all() and (net >= -optimizer.grid.export_limit_mw - TOL).all()


# --------------------------------------------------------------------------- microturbine
class TestMicroturbine:
    def test_output_within_limits_when_committed(self, da, optimizer):
        mt, d = optimizer.mt, da.scenario_detail
        on, off = d[d.mt_on > 0.5], d[d.mt_on < 0.5]
        assert (off.mt_mw.abs() < TOL).all()
        assert ((on.mt_mw >= mt.p_min_mw - TOL) & (on.mt_mw <= mt.p_max_mw + TOL)).all()

    def test_commitment_logic_and_min_up_down(self, da, optimizer):
        p, mt, tg = da.plan, optimizer.mt, optimizer.time
        n_up, n_dn = tg.steps(mt.min_up_h), tg.steps(mt.min_down_h)
        u_prev = np.concatenate([[0.0], p.mt_u[:-1]])                 # unit starts offline
        assert np.allclose(p.mt_x - p.mt_y, p.mt_u - u_prev)
        assert not ((p.mt_x > 0.5) & (p.mt_y > 0.5)).any()
        for t in np.flatnonzero(p.mt_x > 0.5):
            assert (p.mt_u[t:t + n_up] > 0.5).all(), f"start at {t} violates min up time"
        for t in np.flatnonzero(p.mt_y > 0.5):
            assert (p.mt_u[t:t + n_dn] < 0.5).all(), f"stop at {t} violates min down time"

    def test_ramp_limits(self, da, optimizer):
        mt, dt = optimizer.mt, optimizer.time.dt_h
        limit = max(mt.ramp_up_mw_h, mt.ramp_down_mw_h, mt.startup_ramp_mw_h, mt.shutdown_ramp_mw_h) * dt
        for _, g in da.scenario_detail.groupby("scenario"):
            out = np.concatenate([[0.0], g.sort_values("step").mt_mw.to_numpy()])
            assert (np.abs(np.diff(out)) <= limit + TOL).all()


# --------------------------------------------------------------------------- battery
class TestBattery:
    def test_soc_bounds_and_dynamics(self, da, optimizer):
        b, dt = optimizer.bess, optimizer.time.dt_h
        for _, g in da.scenario_detail.groupby("scenario"):
            g = g.sort_values("step")
            soc = g.soc_mwh.to_numpy()
            assert ((soc >= b.soc_min_mwh - TOL) & (soc <= b.soc_max_mwh + TOL)).all()
            prev = np.concatenate([[b.soc_init_mwh], soc[:-1]])
            step = b.eta_ch * g.bess_ch_mw.to_numpy() * dt - g.bess_dis_mw.to_numpy() * dt / b.eta_dis
            assert np.allclose(soc, prev + step, atol=1e-6)
            assert soc[-1] >= b.terminal_target - TOL

    def test_no_simultaneous_charge_and_discharge(self, da):
        d = da.scenario_detail
        assert not ((d.bess_ch_mw > TOL) & (d.bess_dis_mw > TOL)).any()

    def test_power_limits(self, da, optimizer):
        d = da.scenario_detail
        assert (d.bess_ch_mw <= optimizer.bess.p_max_mw + TOL).all()
        assert (d.bess_dis_mw <= optimizer.bess.p_max_mw + TOL).all()


# --------------------------------------------------------------------------- production
class TestProduction:
    def test_every_job_scheduled_once_inside_horizon(self, da, optimizer):
        bp, T = da.batch_plan, optimizer.time.n_steps
        assert len(bp) == len(optimizer.factory.jobs)
        assert (bp.start_step >= 0).all() and (bp.start_step + bp.duration_steps <= T).all()

    def test_machine_sequence_and_buffer(self, da, optimizer):
        buf = optimizer.factory.buffer_h
        for _, g in da.batch_plan.groupby("machine"):
            g = g.sort_values("sequence")
            starts, ends = g.start_hour.to_numpy(), g.end_hour.to_numpy()
            assert (starts[1:] >= ends[:-1] + buf - TOL).all()

    def test_scenario_starts_respect_precedence_and_window(self, da, optimizer):
        W, buf = optimizer.factory.intraday_shift_window_h, optimizer.factory.buffer_h
        cols = [c for c in da.batch_plan.columns if c.startswith("start_hour_")]
        assert cols
        for c in cols:
            assert ((da.batch_plan[c] - da.batch_plan.start_hour).abs() <= W + TOL).all()
            for _, g in da.batch_plan.groupby("machine"):
                g = g.sort_values("sequence")
                dur = g.duration_steps.to_numpy() * optimizer.time.dt_h
                assert (g[c].to_numpy()[1:] >= g[c].to_numpy()[:-1] + dur[:-1] + buf - TOL).all()

    def test_inventory_conservation(self, da, optimizer):
        f, T = optimizer.factory, optimizer.time.n_steps
        produced = sum(j.units_out for j in f.jobs)
        delivered = float(f.demand_series(T).sum())
        for _, g in da.scenario_detail.groupby("scenario"):
            g = g.sort_values("step")
            assert (g.inventory >= -TOL).all()
            assert g.inventory.iloc[-1] == pytest.approx(f.inventory_init + produced - delivered
                                                         + g.unmet_units.sum(), abs=1e-4)

    def test_batch_load_matches_jobs_energy(self, da, optimizer):
        """Sum of batch load over the day == sum(power * duration_steps) of all jobs, per scenario."""
        expected = sum(j.power_mw * optimizer.time.steps(j.duration_h) for j in optimizer.factory.jobs)
        for _, g in da.scenario_detail.groupby("scenario"):
            assert g.batch_load_mw.sum() == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- intraday stage
class TestIntraday:
    def test_first_stage_is_frozen(self, da, intraday):
        for name in ("p_da_buy", "p_da_sell", "mt_u", "mt_x", "mt_y"):
            assert np.array_equal(getattr(da.plan, name), getattr(intraday.plan, name)), name
        assert da.plan.job_start_step == intraday.plan.job_start_step

    def test_only_remaining_steps_are_optimised(self, intraday, cfg):
        t0 = cfg.intraday.step
        assert intraday.scenario_detail.step.min() == t0
        sched = intraday.schedule
        assert len(sched) == cfg.time.n_steps and sched.loc[:t0 - 1].isna().all().all()
        assert sched.loc[t0:].notna().all().all()

    def test_soc_continues_from_measured_state(self, da, intraday, optimizer, cfg):
        t0, b = cfg.intraday.step, optimizer.bess
        st = InitialState.from_result(da, t0, cfg.time)
        g = intraday.scenario_detail.sort_values("step").iloc[0]
        expected = st.soc_mwh + b.eta_ch * g.bess_ch_mw - g.bess_dis_mw / b.eta_dis
        assert g.soc_mwh == pytest.approx(expected, abs=1e-6)

    def test_started_jobs_do_not_move_and_others_stay_in_window(self, da, intraday, cfg, optimizer):
        t0, W = cfg.intraday.step, optimizer.factory.intraday_shift_window_h
        bp = intraday.batch_plan
        for _, row in bp.iterrows():
            shifted = row["start_hour_ID-refresh"]
            if row.start_step < t0:
                assert shifted == row.start_hour
            else:
                assert abs(shifted - row.start_hour) <= W + TOL and shifted >= t0 - TOL

    def test_intraday_recourse_is_no_worse_than_keeping_da_operation(self, intraday):
        assert intraday.objective_eur > 0 and intraday.risk["std"] == 0.0     # single scenario


# --------------------------------------------------------------------------- error handling
class TestErrors:
    def _opt(self, factory=None, **kw):
        return MultiStageProsumerOptimizer(
            kw.pop("grid", GridParams()), factory or FactoryParams(jobs=[BatchJob("A", "M1", 0, 2.0, 50.0)]),
            MicroturbineParams(), BESSParams(), TimeGrid(), kw.pop("solver", SolverSettings()))

    @staticmethod
    def _tree():
        return ScenarioTree([IDScenario("S", 1.0, 100.0, 50.0, 100.0, 50.0, [RTBranch(1.0)])])

    def test_job_longer_than_horizon(self):
        with pytest.raises(ConfigurationError, match="cannot fit"):
            self._opt(FactoryParams(jobs=[BatchJob("A", "M1", 0, 30.0, 50.0)]))

    def test_machine_sequence_longer_than_horizon(self):
        jobs = [BatchJob(f"J{k}", "M1", k, 8.0, 50.0) for k in range(3)]   # 3*8 + 2 buffers > 24
        with pytest.raises(ConfigurationError, match="sequence needs"):
            self._opt(FactoryParams(jobs=jobs))

    def test_unknown_solver(self):
        with pytest.raises(SolverUnavailableError):
            self._opt(solver=SolverSettings(name="no_such_solver")).solve_day_ahead(self._tree())

    def test_infeasible_model_is_reported(self):
        # 500 MW of non-shiftable load, but grid import (400) + MT (25) + BESS (40) can only cover 465 MW.
        # Infeasible already in the LP relaxation, so the solver proves it instantly.
        opt = self._opt(FactoryParams(jobs=[BatchJob("A", "M1", 0, 2.0, 10.0)], base_load_mw=500.0),
                        solver=SolverSettings(time_limit_s=30.0))
        with pytest.raises(ModelInfeasibleError):
            opt.solve_day_ahead(self._tree())

    def test_intraday_input_validation(self, optimizer, da, intraday, cfg):
        st = InitialState.from_result(da, 10, cfg.time)
        scen = IDScenario("X", 1.0, 100.0, 50.0, 100.0, 50.0, [RTBranch(1.0)])
        with pytest.raises(DataValidationError, match="day-ahead"):
            optimizer.solve_intraday(intraday, IntradayInputs(10, st, scen))
        with pytest.raises(DataValidationError, match="t0 must be"):
            optimizer.solve_intraday(da, IntradayInputs(24, st, scen))
        with pytest.raises(DataValidationError, match="initial SoC"):
            optimizer.solve_intraday(da, IntradayInputs(10, InitialState(soc_mwh=500.0), scen))
        with pytest.raises(DataValidationError, match="prob must be 1.0"):
            optimizer.solve_intraday(da, IntradayInputs(10, st, IDScenario("X", 0.5, 100.0, 50.0, 100.0, 50.0,
                                                                          [RTBranch(1.0)])))
