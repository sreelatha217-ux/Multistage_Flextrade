import copy

import pytest

from prosumer_opt import IDScenario, RTBranch, ScenarioSettings, ScenarioTree, TimeGrid
from prosumer_opt.data import build_demo_tree, tou_prices
from prosumer_opt.exceptions import DataValidationError

T = 4


def scenario(prob=1.0, **kw):
    d = dict(name="S", prob=prob, da_buy=100.0, da_sell=50.0, id_buy=100.0, id_sell=50.0,
             rt_branches=[RTBranch(1.0, 0.0, 1.2, 0.8)])
    d.update(kw)
    return IDScenario(**d)


def test_valid_tree_is_converted_to_arrays():
    out = ScenarioTree([scenario()]).validated(T)
    assert out.scenarios[0].da_buy.shape == (T,)
    assert out.scenarios[0].rt_branches[0].r_minus.shape == (T,)


@pytest.mark.parametrize("bad, msg", [
    (dict(prob=0.5), "probabilities sum"),
    (dict(da_sell=150.0), "da_sell must be <= da_buy"),
    (dict(id_sell=150.0), "id_sell must be <= id_buy"),
    (dict(rt_branches=[]), "at least one RT branch"),
    (dict(rt_branches=[RTBranch(0.5)]), "RT probabilities sum"),
    (dict(rt_branches=[RTBranch(1.0, 0.0, 0.9, 1.1)]), "r_minus must be >= r_plus"),
    (dict(rt_branches=[RTBranch(1.0, -1.5)]), "load_dev must be > -1"),
    (dict(da_buy=[1.0, 2.0]), "expected length"),
])
def test_invalid_trees_are_rejected(bad, msg):
    with pytest.raises(DataValidationError, match=msg):
        ScenarioTree([scenario(**bad)]).validated(T)


def test_empty_tree_rejected():
    with pytest.raises(DataValidationError, match="empty"):
        ScenarioTree([]).validated(T)


def test_validation_does_not_mutate_input():
    tree = ScenarioTree([scenario()])
    before = copy.deepcopy(tree)
    tree.validated(T)
    assert tree.scenarios[0].da_buy == before.scenarios[0].da_buy == 100.0


def test_demo_tree_is_reproducible_and_valid():
    tg, cfg = TimeGrid(), ScenarioSettings(n_id_scenarios=4, n_rt_branches=2)
    a, b = build_demo_tree(tg, cfg), build_demo_tree(tg, cfg)
    assert [float(s.da_buy[5]) for s in a.scenarios] == [float(s.da_buy[5]) for s in b.scenarios]
    a.validated(tg.n_steps)
    assert sum(s.prob for s in a.scenarios) == pytest.approx(1.0)


def test_tou_tariff_matches_reference_table():
    buy, sell = tou_prices(TimeGrid())
    assert (buy[3], sell[3]) == (66.42, 36.53)        # off-peak 02-04
    assert (buy[5], sell[5]) == (88.56, 48.71)        # flat 04-06
    assert (buy[0], sell[0]) == (118.08, 64.94)       # mid 00-02
    assert (buy[9], sell[9]) == (177.12, 97.42)       # on-peak 08-12
    assert (buy > 0).all() and (sell <= buy).all()    # every hour covered, no arbitrage
