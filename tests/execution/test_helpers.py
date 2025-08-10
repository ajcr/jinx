import numpy as np
import pytest


from src.jinx.execution.helpers import (
    maybe_pad_by_duplicating_atoms,
    maybe_pad_with_fill_value,
)


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
    result = maybe_pad_with_fill_value(arrays)
    assert len(result) == len(expected)
    for arr, expected in zip(result, expected):
        np.testing.assert_array_equal(arr, expected)


@pytest.mark.parametrize(
    "arrays, expected",
    [
        pytest.param(
            [np.array([1, 2, 3])],
            [np.array([1, 2, 3])],
            id="[[1, 2, 3]]",
        ),
        pytest.param(
            [1, 2],
            [np.array([1]), np.array([2])],
            id="[1, 2]",
        ),
        pytest.param(
            [np.array([1]), 2],
            [np.array([1]), np.array([2])],
            id="[[1], 2]",
        ),
        pytest.param(
            [np.array(1), np.array([1, 2, 3])],
            [np.array([1, 1, 1]), np.array([1, 2, 3])],
            id="[1, [1, 2, 3]]",
        ),
        pytest.param(
            [np.array([1]), np.array([1, 2, 3])],
            [np.array([1, 0, 0]), np.array([1, 2, 3])],
            id="[[1], [1, 2, 3]]",
        ),
        pytest.param(
            [np.array([[1, 2], [3, 4]]), np.array([7])],
            [np.array([[1, 2], [3, 4]]), np.array([[7, 0], [0, 0]])],
            id="[[1, 2], [3, 4]], [7]",
        ),
        pytest.param(
            [np.array([[1, 2], [3, 4]]), np.array(8)],
            [np.array([[1, 2], [3, 4]]), np.array([[8, 8], [8, 8]])],
            id="[[1, 2], [3, 4]], 8",
        ),
    ],
)
def test_maybe_pad_by_duplicating_atoms(arrays, expected):
    result = maybe_pad_by_duplicating_atoms(arrays, ignore_first_dim=False)
    assert len(result) == len(expected)
    for arr, exp in zip(result, expected):
        np.testing.assert_array_equal(arr, exp)


@pytest.mark.parametrize(
    "arrays, expected",
    [
        pytest.param(
            [np.array([1, 2, 3])],
            [np.array([1, 2, 3])],
            id="[[1, 2, 3]]",
        ),
        pytest.param(
            [np.array([1]), 2],
            [np.array([1]), np.array([2])],
            id="[[1], 2]",
        ),
        pytest.param(
            [1, 2],
            [np.array([1]), np.array([2])],
            id="[1, 2]",
        ),
        pytest.param(
            [np.array(1), np.array([1, 2, 3])],
            [np.array([1, 1, 1]), np.array([1, 2, 3])],
            id="[1, [1, 2, 3]]",
        ),
        pytest.param(
            [np.array([1]), np.array([1, 2, 3])],
            [np.array([1, 0, 0]), np.array([1, 2, 3])],
            id="[[1], [1, 2, 3]]",
        ),
        pytest.param(
            [np.array([[1, 2], [3, 4]]), np.array([7])],
            [np.array([[1, 2], [3, 4]]), np.array([[7, 0]])],
            id="[[1, 2], [3, 4]], [7]",
        ),
        pytest.param(
            [np.array([[1, 2], [3, 4]]), np.array(8)],
            [np.array([[1, 2], [3, 4]]), np.array([[8, 8]])],
            id="[[1, 2], [3, 4]], 8",
        ),
    ],
)
def test_maybe_pad_by_duplicating_atoms_ignore_first_dim(arrays, expected):
    result = maybe_pad_by_duplicating_atoms(arrays, ignore_first_dim=True)
    assert len(result) == len(expected)
    for arr, exp in zip(result, expected):
        np.testing.assert_array_equal(arr, exp)
