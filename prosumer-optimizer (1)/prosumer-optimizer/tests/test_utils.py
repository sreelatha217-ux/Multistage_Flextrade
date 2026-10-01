import numpy as np
import pytest

from prosumer_opt.exceptions import ConfigurationError, DataValidationError
from prosumer_opt.utils import as_series, cvar, require


def test_require_raises_given_exception():
    require(True, "fine")
    with pytest.raises(ConfigurationError, match="boom"):
        require(False, "boom")
    with pytest.raises(DataValidationError):
        require(False, "x", DataValidationError)


def test_as_series_broadcasts_and_validates():
    assert np.array_equal(as_series(2.5, 3, "a"), [2.5, 2.5, 2.5])
    assert np.array_equal(as_series([1, 2, 3], 3, "a"), [1, 2, 3])
    with pytest.raises(DataValidationError, match="expected length 3"):
        as_series([1, 2], 3, "a")
    with pytest.raises(DataValidationError, match="NaN"):
        as_series([1, np.nan, 3], 3, "a")
    with pytest.raises(DataValidationError, match="cannot convert"):
        as_series("abc", 3, "a")


def test_cvar_uniform_distribution():
    costs, probs = [10, 20, 30, 40], [0.25] * 4
    assert cvar(costs, probs, 0.5) == pytest.approx(35.0)      # mean of worst two
    assert cvar(costs, probs, 0.75) == pytest.approx(40.0)     # worst one
    assert cvar(costs, probs, 1.0) == 40.0                      # alpha = 1 -> worst case
    assert cvar(costs, probs, 0.0) == pytest.approx(25.0)       # alpha = 0 -> mean
