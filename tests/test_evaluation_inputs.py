"""Invalid coordinates must not silently reshape into different ground truth."""

import numpy as np
import pytest

from ionrecog.evaluation import match_points


@pytest.mark.parametrize("points", [
    [1, 2],
    [1, 2, 3, 4],
    [[1, 2, 3, 4]],
    [[], []],
    np.empty((0, 3)),
    [[[1, 2]]],
    [[1, 2], [3]],
    0,
    None,
])
@pytest.mark.parametrize("argument", ["predicted", "truth"])
def test_coordinate_rows_must_explicitly_have_two_columns(points, argument):
    values = {"predicted": [[1, 2]], "truth": [[1, 2]]}
    values[argument] = points
    with pytest.raises(ValueError, match=argument + " must be a finite Nx2"):
        match_points(**values)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("argument", ["predicted", "truth"])
def test_nonfinite_coordinates_are_rejected_even_against_empty_set(value, argument):
    values = {"predicted": [], "truth": []}
    values[argument] = [[value, 1]]
    with pytest.raises(ValueError, match=argument + " must be a finite Nx2"):
        match_points(**values)


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, np.inf, -np.inf, None, [1]])
def test_tolerance_must_be_finite_positive_scalar(tolerance):
    with pytest.raises(ValueError, match="tolerance must be a finite positive scalar"):
        match_points([], [], tolerance=tolerance)


@pytest.mark.parametrize("points", [[], np.empty((0, 2))])
def test_empty_sets_remain_valid(points):
    result = match_points(points, points, tolerance=1)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 0)
    assert result["all_correct"]
    assert result["mean_error_px"] is None
