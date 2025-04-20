import numpy as np
import pytest

from src.jinx.execution.application import maybe_pad_with_fill_value


@pytest.mark.parametrize(
    "arrays, expected",
    [
        pytest.param(
            [np.array([1, 2, 3])],
            [np.array([1, 2, 3])],
            id="[1, 2, 3]",
        ),
        pytest.param(
            [np.array([1, 2, 3]), np.array([7])],
            [np.array([1, 2, 3]), np.array([7, 0, 0])],
            id="[1, 2, 3], [7]",
        ),
        pytest.param(
            [np.array([[1, 2], [3, 4]]), np.array([[7]])],
            [np.array([[1, 2], [3, 4]]), np.array([[7, 0], [0, 0]])],
            id="[[1, 2], [3, 4]], [[7]]",
        ),
        pytest.param(
            [np.array([[1, 2], [3, 4]]), np.array([[7], [8]])],
            [np.array([[1, 2], [3, 4]]), np.array([[7, 0], [8, 0]])],
            id="[[1, 2], [3, 4]], [[7], [8]]",
        ),
    ],
)
def test_maybe_pad_with_fill_value(arrays, expected):
    result = maybe_pad_with_fill_value(arrays, fill_value=0)
    assert [np.array_equal(arr, expected) for arr in result]
