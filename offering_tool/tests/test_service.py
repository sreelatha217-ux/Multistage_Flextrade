"""Tests for the service layer. They need no solver: the model run is replaced by saved results."""
from __future__ import annotations

import copy
import datetime as dt
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from offering_service import db_market, mapper, output_builder, runner, validate     # noqa: E402
from offering_service.errors import DataUnavailable, InvalidInput                    # noqa: E402

EX = ROOT / "examples"
SAMPLE = ROOT / "sample_results"
TZ = ZoneInfo("Europe/Copenhagen")


def load(name):
    return json.loads((EX / name).read_text())


def defined_flags() -> set[str]:
    """Every --flag the model's own parsers define (read from the source, no import needed)."""
    flags = set()
    for f in ("factory_mt_da_scheduler.py", "factory_mt_id_scheduler.py", "factory_mt_rt_scheduler.py",
              "factory_mt_offering_cli.py"):
        flags |= set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', (ROOT / f).read_text()))
    return flags


class Validation(unittest.TestCase):
    def test_examples_pass(self):
        self.assertEqual(validate.validate_input(load("input_generate.json")), [])
        validate.validate_input(load("input_database.json"))

    def bad(self, mutate, field):
        d = load("input_generate.json")
        mutate(d)
        with self.assertRaises(InvalidInput) as cm:
            validate.validate_input(d)
        self.assertEqual(cm.exception.field, field)

    def test_field_names_in_errors(self):
        self.bad(lambda d: d["plant"].update(duration_h=[[0.0] * 6] * 3), "plant.duration_h")
        self.bad(lambda d: d["grid"]["reference_tariff"]["sell_eur_mwh"].__setitem__(3, 999), "grid.reference_tariff.sell_eur_mwh")
        self.bad(lambda d: d["bess"].update(soc_init_mwh=500), "bess.soc_init_mwh")
        self.bad(lambda d: d["microturbine"].update(block_cost_eur_mwh=[50, 40, 60]), "microturbine.block_cost_eur_mwh")
        self.bad(lambda d: d["run_settings"].update(risk_aversion_beta=2), "run_settings.risk_aversion_beta")
        self.bad(lambda d: d["run_settings"].update(start_step_h=0.7), "run_settings.horizon_h")
        self.bad(lambda d: d["plant"].update(demand_units=[100.0] * 24), "plant.demand_units")
        self.bad(lambda d: d["reserve_flex"].update(assets=["mt", "wind"]), "reserve_flex.assets")
        self.bad(lambda d: d["plant"].update(machines=["m1", "m1", "m3"]), "plant.machines")

    def test_microturbine_null_and_bess_null_are_valid(self):
        d = load("input_generate.json")
        d["microturbine"], d["bess"] = None, None
        validate.validate_input(d)

    def test_provided_negative_price_rejected(self):
        d = load("input_generate.json")
        S, T = 2, 24
        flat = lambda v: [[v] * T for _ in range(S)]
        d["market_scenarios"] = dict(mode="provided", prob=[0.5, 0.5], da_buy=flat(50.0), da_sell=flat(30.0),
                                     sr_up=flat(5.0), id_buy=flat(50.0), id_sell=flat(30.0), load_dev_mw=flat(0.0))
        d["realtime_scenarios"] = dict(mode="generate")
        validate.validate_input(d)
        d["market_scenarios"]["da_buy"][0][3] = -4.0
        with self.assertRaises(InvalidInput) as cm:
            validate.validate_input(d)
        self.assertEqual(cm.exception.field, "market_scenarios.da_buy")


class Mapping(unittest.TestCase):
    def test_every_flag_exists_in_the_model(self):
        known = defined_flags()
        for name in ("input_generate.json", "input_database.json"):
            d = load(name)
            if name.endswith("database.json"):
                d["market_scenarios"] = dict(mode="generate")          # no database in this test
                d["grid"]["reference_tariff"] = dict(buy_eur_mwh=[60.0] * 24, sell_eur_mwh=[30.0] * 24)
            with tempfile.TemporaryDirectory() as tmp:
                argv, _ = mapper.build_files(d, Path(tmp))
            used = {a for a in argv if a.startswith("--")}
            self.assertEqual(used - known, set(), f"unknown flags in {name}")

    def test_plant_file_matches_model_format(self):
        d = load("input_generate.json")
        with tempfile.TemporaryDirectory() as tmp:
            mapper.build_files(d, Path(tmp))
            inst = json.loads((Path(tmp) / "plant.json").read_text())
        for k in ("machines", "tasks", "power_mw", "duration_h", "yield_units", "base_load_mw",
                  "price_buy_eur_mwh", "price_sell_eur_mwh", "demand_units", "horizon_h", "buffer_h",
                  "grid_limit_mw", "grid_sell_limit_mw", "max_batches_per_machine", "inventory_init",
                  "inventory_max", "mt", "bess"):
            self.assertIn(k, inst)
        self.assertEqual(inst["bess"]["degradation_eur_mwh"], 5)

    def test_mt_null_and_default(self):
        d = load("input_generate.json")
        d["microturbine"] = None
        with tempfile.TemporaryDirectory() as tmp:
            mapper.build_files(d, Path(tmp))
            self.assertIsNone(json.loads((Path(tmp) / "plant.json").read_text())["mt"])
        del d["microturbine"]
        with tempfile.TemporaryDirectory() as tmp:
            mapper.build_files(d, Path(tmp))
            self.assertNotIn("mt", json.loads((Path(tmp) / "plant.json").read_text()))   # model default MT

    def test_unknown_microturbine_field_rejected(self):
        d = load("input_generate.json")
        d["microturbine"]["turbo"] = 1
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(InvalidInput):
            mapper.build_files(d, Path(tmp))

    def test_reserve_asset_names_match_model(self):
        src = (ROOT / "factory_mt_offering_strategy.py").read_text()
        m = re.search(r'^ASSETS\s*=\s*\(([^)]*)\)', src, re.M)
        self.assertEqual(tuple(re.findall(r'"([a-z_]+)"', m.group(1))), validate.ASSETS)

    def test_mt_and_bess_keys_match_model_dataclasses(self):
        src = (ROOT / "factory_mt_da_scheduler.py").read_text()
        for cls, keys in (("Microturbine", mapper.MT_KEYS), ("BESS", mapper.BESS_KEYS)):
            body = src.split(f"class {cls}:")[1].split("def ")[0].split("@property")[0]
            fields = set(re.findall(r"^    ([a-z_]+)\s*:", body, re.M))
            self.assertEqual(fields, keys, cls)


class Output(unittest.TestCase):
    def test_build_from_saved_results(self):
        out = output_builder.build_output(SAMPLE, include_scenario_detail=True, requested_mip_gap=0.01)
        summ = json.loads((SAMPLE / "summary.json").read_text())
        self.assertEqual(out["meta"]["status"], summ["status"])
        self.assertAlmostEqual(out["summary"]["expected_profit_eur"], summ["kpis"]["exp_profit_eur"])
        self.assertEqual(len(out["day_ahead"]["hourly_offers"]), 24)
        h0 = out["day_ahead"]["hourly_offers"][0]
        self.assertIn("price_eur_mw_h", h0["reserve_offer_up"][0])
        self.assertNotIn("price_eur_mwh", h0["reserve_offer_up"][0])
        self.assertTrue(out["day_ahead"]["production_schedule"])
        self.assertEqual(len(out["day_ahead"]["expected_position"]), 24)
        self.assertEqual(len(out["real_time"]["hourly_expected"]), 24)
        self.assertGreaterEqual(len(out["intraday"]["reschedules"]), 1)
        self.assertTrue(out["intraday"]["reschedules"][0]["jobs"])
        self.assertTrue(out["real_time"]["scenario_detail"])
        self.assertTrue(out["strategy_comparison"]["strategies"])
        self.assertIsNotNone(out["scenarios_used"]["price"])
        json.dumps(out, allow_nan=False)                     # strict JSON: no NaN anywhere

    def test_scenario_detail_off_by_default(self):
        self.assertNotIn("scenario_detail", output_builder.build_output(SAMPLE)["real_time"])


def fake_rows(day: dt.date, col: str, fn, hours=range(24)):
    base = dt.datetime(day.year, day.month, day.day, tzinfo=TZ)
    rows = []
    for h in hours:
        t = (base.astimezone(dt.timezone.utc) + dt.timedelta(hours=h))
        rows.append({"delivery_hour_start": t, "area": "DK1", col: fn(h)})
    return rows


class FakeConn:
    def __init__(self, da, sr):
        self.da, self.sr = da, sr

    def cursor(self):
        return self

    def execute(self, text, params):
        self._rows = self.da if "da_price_eur_mwh" in text else self.sr
        self.description = [(k,) for k in self._rows[0]] if self._rows else [("x",)]
        assert set(params) == {"area", "first_day", "last_day"}

    def fetchall(self):
        return [tuple(r.values()) for r in self._rows]

    def close(self):
        pass


class Database(unittest.TestCase):
    ms = dict(mode="database", area="DK1", buy_adder_eur_mwh=20.0, sell_deduct_eur_mwh=5.0, id_spread_sigma=0.0)

    def conn(self, days, da_fn=lambda h: 50.0 + h, sr_fn=lambda h: 8.0, drop=None):
        da, sr = [], []
        for d in days:
            hrs = [h for h in range(24) if not (drop and d == drop[0] and h == drop[1])]
            da += fake_rows(d, "da_price_eur_mwh", da_fn, hrs)
            sr += fake_rows(d, "mfrr_up_capacity_price_eur_mw_h", sr_fn)
        return FakeConn(da, sr)

    def test_builds_scenarios_and_rule(self):
        days = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(3)]
        scen, tariff, info = db_market.build_scenarios({**self.ms, "from": "2026-09-01", "to": "2026-09-03"},
                                                      24, self.conn(days))
        self.assertEqual(len(scen["prob"]), 3)
        self.assertAlmostEqual(sum(scen["prob"]), 1.0)
        self.assertAlmostEqual(scen["da_buy"][0][0], 70.0)          # 50 + 20
        self.assertAlmostEqual(scen["da_sell"][0][0], 45.0)         # 50 - 5
        self.assertAlmostEqual(scen["sr_up"][0][5], 8.0)
        self.assertEqual(scen["id_buy"], scen["da_buy"])            # sigma 0 -> ID equals DA
        self.assertEqual(len(tariff["buy_eur_mwh"]), 24)
        self.assertEqual(info["days_used"], ["2026-09-01", "2026-09-02", "2026-09-03"])

    def test_negative_prices_are_floored_and_reported(self):
        d = [dt.date(2026, 9, 1)]
        scen, _, info = db_market.build_scenarios({**self.ms, "days": ["2026-09-01"]}, 24,
                                                  self.conn(d, da_fn=lambda h: -30.0 if h == 4 else 40.0))
        self.assertEqual(scen["da_buy"][0][4], 0.0)
        self.assertEqual(scen["da_sell"][0][4], 0.0)
        self.assertTrue(all(s <= b for s, b in zip(scen["da_sell"][0], scen["da_buy"][0])))
        self.assertGreater(info["negative_values_floored"], 0)
        self.assertTrue(info["warnings"])

    def test_incomplete_and_clock_change_days_are_dropped(self):
        days = [dt.date(2026, 9, 1), dt.date(2026, 9, 2)]
        scen, _, info = db_market.build_scenarios({**self.ms, "from": "2026-09-01", "to": "2026-09-02"}, 24,
                                                  self.conn(days, drop=(days[1], 7)))
        self.assertEqual(info["days_used"], ["2026-09-01"])
        self.assertIn("2026-09-02", info["days_dropped"])
        # 25-hour day (clock goes back on 25 Oct 2026) has a repeated local hour
        d = dt.date(2026, 10, 25)
        base = dt.datetime(2026, 10, 25, tzinfo=TZ).astimezone(dt.timezone.utc)
        da = [{"delivery_hour_start": base + dt.timedelta(hours=h), "area": "DK1", "da_price_eur_mwh": 40.0}
              for h in range(25)]
        sr = fake_rows(d, "mfrr_up_capacity_price_eur_mw_h", lambda h: 5.0)
        with self.assertRaises(DataUnavailable):
            db_market.build_scenarios({**self.ms, "days": ["2026-10-25"]}, 24, FakeConn(da, sr))

    def test_n_scenarios_selection(self):
        days = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(10)]
        _, _, info = db_market.build_scenarios({**self.ms, "from": "2026-09-01", "to": "2026-09-10", "n_scenarios": 3},
                                               24, self.conn(days))
        self.assertEqual(len(info["days_used"]), 3)
        self.assertEqual(info["days_used"][0], "2026-09-01")
        self.assertEqual(info["days_used"][-1], "2026-09-10")
        with self.assertRaises(DataUnavailable):
            db_market.build_scenarios({**self.ms, "from": "2026-09-01", "to": "2026-09-10", "n_scenarios": 12},
                                      24, self.conn(days))

    def test_missing_env_gives_data_unavailable(self):
        import os
        saved = {k: os.environ.pop(k, None) for k in ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD")}
        try:
            with self.assertRaises(DataUnavailable):
                db_market.connect()
        finally:
            os.environ.update({k: v for k, v in saved.items() if v})

    def test_scenarios_pass_model_rules(self):
        days = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(4)]
        scen, tariff, _ = db_market.build_scenarios({**self.ms, "id_spread_sigma": 0.08, "days": [d.isoformat() for d in days]},
                                                    24, self.conn(days, da_fn=lambda h: -5.0 + 10 * (h % 5)))
        for k in ("da_buy", "da_sell", "sr_up", "id_buy", "id_sell"):
            self.assertTrue(all(v >= 0 for row in scen[k] for v in row), k)
        for a, b in (("da_sell", "da_buy"), ("id_sell", "id_buy")):
            self.assertTrue(all(s <= bb + 1e-9 for rs, rb in zip(scen[a], scen[b]) for s, bb in zip(rs, rb)))
        self.assertTrue(all(s <= b + 1e-9 for s, b in zip(tariff["sell_eur_mwh"], tariff["buy_eur_mwh"])))


class OwnerRun(unittest.TestCase):
    """The saved owner run (seed 2024, 30 tasks, 3 x 3 scenarios) as a complete input.json / output.json pair."""

    def test_input_is_valid_and_maps_back_to_the_original_files(self):
        d = load("input_owner_run.json")
        self.assertEqual(validate.validate_input(d), [])
        with tempfile.TemporaryDirectory() as tmp:
            mapper.build_files(d, Path(tmp))
            plant = json.loads((Path(tmp) / "plant.json").read_text())
            scen = json.loads((Path(tmp) / "scenario_set.json").read_text())
        orig = json.loads((EX / "owner_run" / "plant.json").read_text())
        self.assertEqual({k: plant[k] for k in orig}, orig)
        self.assertEqual(scen, json.loads((SAMPLE / "scenario_set.json").read_text()))
        self.assertEqual(len(d["plant"]["tasks"]), 30)

    def test_saved_jobs_use_the_plant_numbers(self):
        d = load("input_owner_run.json")
        P = d["plant"]
        for j in output_builder.read_csv(SAMPLE / "jobs.csv"):
            m, p = P["machines"].index(j["machine"]), P["tasks"].index(j["task"])
            self.assertAlmostEqual(P["power_mw"][m][p], j["power_mw"])
            self.assertAlmostEqual(P["duration_h"][m][p], j["duration_h"])

    def test_output_example_is_current(self):
        saved = json.loads((EX / "output_owner_run.json").read_text())
        fresh = output_builder.build_output(SAMPLE, False, 0.01)
        self.assertEqual(saved, json.loads(json.dumps(fresh)))


class EndToEnd(unittest.TestCase):
    """Whole pipeline with the model replaced by saved results."""

    def run_job(self, d, fake_model=True):
        tmp = Path(tempfile.mkdtemp())
        inp, outp = tmp / "input.json", tmp / "output.json"
        inp.write_text(json.dumps(d))
        orig = runner.run_model

        def fake(argv, timeout):
            out = Path(argv[argv.index("--out") + 1])
            shutil.copytree(SAMPLE, out)

        runner.run_model = fake if fake_model else orig
        try:
            rc = runner.run(inp, outp, tmp / "work")
        finally:
            runner.run_model = orig
        return rc, json.loads(outp.read_text()), tmp

    def test_success(self):
        rc, out, tmp = self.run_job(load("input_generate.json"))
        self.assertEqual(rc, 0)
        self.assertEqual(out["meta"]["status"], "optimal")
        self.assertEqual(json.loads((tmp / "work" / "status.json").read_text())["state"], "done")

    def test_invalid_input_writes_error_file(self):
        d = load("input_generate.json")
        d["bess"]["soc_init_mwh"] = 999
        rc, out, _ = self.run_job(d)
        self.assertEqual(rc, 1)
        self.assertEqual(out["meta"]["status"], "error")
        self.assertEqual(out["meta"]["error"]["code"], "INVALID_INPUT")
        self.assertEqual(out["meta"]["error"]["field"], "bess.soc_init_mwh")

    def test_database_unreachable_gives_data_unavailable(self):
        import os
        for k in ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"):
            os.environ.pop(k, None)
        rc, out, _ = self.run_job(load("input_database.json"))
        self.assertEqual(out["meta"]["error"]["code"], "DATA_UNAVAILABLE")

    def test_unreadable_input(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "in.json").write_text("{not json")
        rc = runner.run(tmp / "in.json", tmp / "out.json", tmp / "w")
        self.assertEqual(rc, 1)
        self.assertEqual(json.loads((tmp / "out.json").read_text())["meta"]["error"]["code"], "INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()
