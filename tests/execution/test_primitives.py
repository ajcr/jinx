import pytest
import numpy as np

from src.jinx.execution.conversion import box_dtype
from src.jinx.execution.verbs import (
    comma_dyad,
    dollar_dyad,
    dollar_monad,
    gt_monad,
    lt_monad,
    slashco_monad,
    bslashco_monad,
    numberco_monad,
    commaco_dyad,
)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        pytest.param(np.array([1]), np.array([2]), np.array([1, 2]), id="(1),(1)->(2)"),
        pytest.param(
            np.array([0, 1]),
            np.array([2, 3]),
            np.array([0, 1, 2, 3]),
            id="(0 1),(2 3)",
        ),
        pytest.param(
            np.array([1]), np.array([3, 4]), np.array([1, 3, 4]), id="(1),(2)->(3)"
        ),
        pytest.param(
            np.array([9]),
            np.array([[0, 1], [2, 3]]),
            np.array([[9, 9], [0, 1], [2, 3]]),
            id="(9),(i.2 2)",
        ),
        pytest.param(
            np.array([[0, 1], [2, 3]]),
            np.array([9]),
            np.array([[0, 1], [2, 3], [9, 9]]),
            id="(i.2 2),(9)",
        ),
        pytest.param(
            np.array([7, 8]),
            np.array([[1, 2], [3, 4]]),
            np.array([[7, 8], [1, 2], [3, 4]]),
            id="(7,8),(i.2 2)",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            np.array([7, 8]),
            np.array([[1, 2], [3, 4], [7, 8]]),
            id="(i.2 2),(7,8)",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            np.arange(9).reshape(3, 3),
            np.array([[1, 2, 0], [3, 4, 0], [0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            id="(i.2 2),(i.3 3)",
        ),
        pytest.param(
            np.arange(9).reshape(3, 3),
            np.array([[1, 2], [3, 4]]),
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [1, 2, 0], [3, 4, 0]]),
            id="(i.3 3),(i.2 2)",
        ),
        pytest.param(
            np.arange(8).reshape(2, 4),
            np.arange(9).reshape(3, 3),
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]]
            ),
            id="(i.2 4),(i.3 3)",
        ),
        pytest.param(
            np.arange(9).reshape(3, 3),
            np.arange(8).reshape(2, 4),
            np.array(
                [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0], [0, 1, 2, 3], [4, 5, 6, 7]]
            ),
            id="(i.3 3),(i.2 4)",
        ),
        pytest.param(
            np.arange(4).reshape(2, 2),
            np.arange(9).reshape(1, 3, 3),
            np.array(
                [[[0, 1, 0], [2, 3, 0], [0, 0, 0]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]]]
            ),
            id="(i.2 2),(i.1 3 3)",
        ),
        pytest.param(
            np.arange(9).reshape(1, 3, 3),
            np.arange(4).reshape(2, 2),
            np.array(
                [[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 1, 0], [2, 3, 0], [0, 0, 0]]]
            ),
            id="(i.1 3 3),(i.2 2)",
        ),
        pytest.param(
            np.arange(6),
            np.arange(4).reshape(2, 2),
            np.array([[0, 1, 2, 3, 4, 5], [0, 1, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0]]),
            id="(i.6),(i.2 2)",
        ),
        pytest.param(
            np.arange(4).reshape(2, 2),
            np.arange(6),
            np.array([[0, 1, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]]),
            id="(i.2 2),(i.6)",
        ),
        pytest.param(
            np.array([5]).reshape(1, 1),
            np.arange(3),
            np.array([[5, 0, 0], [0, 1, 2]]),
            id="(1 1 $ 5),(i.3)",
        ),
        pytest.param(
            np.arange(3),
            np.array([5]).reshape(1, 1),
            np.array([[0, 1, 2], [5, 0, 0]]),
            id="(i.3),(1 1 $ 5)",
        ),
    ],
)
def test_comma_dyad(x, y, expected):
    result = comma_dyad(x, y)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "x, y, expected",
    [
        pytest.param(np.array(0), np.array(5), np.array([]), id="0 $ 5"),
        pytest.param(np.array(1), np.array(5), np.array([5]), id="1 $ 5"),
        pytest.param(np.array(2), np.array(5), np.array([5, 5]), id="2 $ 5"),
        pytest.param(
            np.array([2, 2]), np.array(5), np.array([[5, 5], [5, 5]]), id="2 2 $ 5"
        ),
        pytest.param(
            np.array(5), np.array([0, 1]), np.array([0, 1, 0, 1, 0]), id="5 $ 0 1"
        ),
        pytest.param(np.array(2), np.array([0, 1, 2]), np.array([0, 1]), id="2 $ 0 1"),
    ],
)
def test_dollar_dyad(x, y, expected):
    result = dollar_dyad(x, y)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "y, expected",
    [
        pytest.param(np.int64(99), 0, id="$ 99"),
        pytest.param(np.array(99), 0, id="$ 99"),
        pytest.param(np.array([99]), np.array([1]), id="$ 1 $ 99"),
        pytest.param(np.array([[99]]), np.array([1, 1]), id="$ 1 1 $ 99"),
        pytest.param(np.array([-1, 1, 0]), np.array([3]), id="$ -1 1 0"),
        pytest.param(
            np.array([[1, 2], [3, 4]]), np.array([2, 2]), id="$ 2 2 $ 1 2 3 4"
        ),
    ],
)
def test_dollar_monad(y, expected):
    result = dollar_monad(y)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    ["y", "expected"],
    [
        pytest.param(np.array(1), np.array([0]), id="/: 1"),
        pytest.param(np.array([1, 1]), np.array([0, 1]), id="/: 1 2"),
        pytest.param(np.array([1, 2]), np.array([0, 1]), id="/: 1 2"),
        pytest.param(np.array([3, 2, 1]), np.array([2, 1, 0]), id="/: 3 2 1"),
        pytest.param(
            np.array([3, 1, 4, 1, 5, 9]),
            np.array([1, 3, 0, 2, 4, 5]),
            id="/: 3 1 4 1 5 9",
        ),
        pytest.param(
            np.array([0, 1, 0, 0, 0, 0, 1, 0, 0]).reshape(3, 3),
            np.array([1, 0, 2]),
            id="/: (3 3 $ 0 1 0 0 0 0 1 0 0)",
        ),
    ],
)
def test_slashco_monad(y, expected):
    result = slashco_monad(y)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    ["y", "expected"],
    [
        pytest.param(np.array(1), np.array([0]), id="\\: 1"),
        pytest.param(np.array([1, 1]), np.array([0, 1]), id="\\: 1 2"),
        pytest.param(np.array([1, 2]), np.array([1, 0]), id="\\: 1 2"),
        pytest.param(np.array([3, 2, 1]), np.array([0, 1, 2]), id="\\: 3 2 1"),
        pytest.param(
            np.array([3, 1, 4, 1, 5, 9]),
            np.array([5, 4, 2, 0, 1, 3]),
            id="\\: 3 1 4 1 5 9",
        ),
        pytest.param(
            np.array([0, 1, 0, 0, 0, 0, 1, 0, 0]).reshape(3, 3),
            np.array([2, 0, 1]),
            id="\\: (3 3 $ 0 1 0 0 0 0 1 0 0)",
        ),
    ],
)
def test_bslashco_monad(y, expected):
    result = bslashco_monad(y)
    assert np.array_equal(result, expected)


BOX_1 = np.array([(1,)], dtype=box_dtype).squeeze()
BOX_2 = np.array([(np.array([1, 2, 3]),)], dtype=box_dtype).squeeze()


@pytest.mark.parametrize(
    "y, expected",
    [
        pytest.param(np.array(1), BOX_1, id="< 1"),
        pytest.param(np.array([1, 2, 3]), BOX_2, id="< 1 2 3"),
    ],
)
def test_lt_monad_box_containing_integer_array(y, expected):
    result = lt_monad(y)
    assert result.shape == expected.shape == ()
    np.testing.assert_array_equal(result.item(), expected.item(), strict=True)


def test_lt_monad_box_containing_box():
    result = lt_monad(BOX_1)
    assert result.shape == ()
    assert result.dtype == box_dtype
    np.testing.assert_array_equal(result.item()[0], BOX_1, strict=True)


def test_gt_monad_unboxed_array():
    y = np.array([1, 2, 3])
    result = gt_monad(y)
    np.testing.assert_array_equal(result, y, strict=True)


def test_gt_monad_boxed_integer():
    result = gt_monad(BOX_1)
    assert result == BOX_1.item()


def test_gt_monad_boxed_array():
    result = gt_monad(BOX_2)
    np.testing.assert_array_equal(result, BOX_2.item()[0], strict=True)


def test_gt_monad_boxed_array_list():
    # 1 ; 2 3 ; 4 5 6
    boxed_list = np.array(
        [
            (np.array([1]),),
            (np.array([2, 3]),),
            (np.array([4, 5, 6]),),
        ],
        dtype=box_dtype,
    )

    result = gt_monad(boxed_list)
    expected = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])

    np.testing.assert_array_equal(result, expected, strict=True)


def test_comma_dyad_joins_boxes():
    result = comma_dyad(BOX_1, BOX_2)
    expected = np.array([(BOX_1.item()[0],), (BOX_2.item()[0],)], dtype=box_dtype)
    assert result.shape == expected.shape == (2,)
    assert result.dtype == box_dtype
    assert result[0][0] is expected[0].item()[0]
    assert result[1][0] is expected[1].item()[0]


@pytest.mark.parametrize(
    ["y", "expected"],
    [
        pytest.param(np.array(0), np.array([0]), id="#: 0"),
        pytest.param(np.array(-2), np.array([1, 0]), id="#: _2"),
        pytest.param(np.array(5), np.array([1, 0, 1]), id="#: 5"),
        pytest.param(np.array([4]), np.array([1, 0, 0]), id="#: 4"),
        pytest.param(np.array(-7), np.array([0, 0, 1]), id="#: _7"),
        pytest.param(np.array(-8.1), np.array([0, 1, 1, 1.9]), id="#: _8.1"),
        pytest.param(np.array(5.32), np.array([1, 0, 1.32]), id="#: 5.32"),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([[0, 1], [1, 0], [1, 1]]),
            id="#: 1 2 3",
        ),
        pytest.param(
            np.array([[3, -6], [0, 9]]),
            np.array([[[0, 0, 1, 1], [1, 0, 1, 0]], [[0, 0, 0, 0], [1, 0, 0, 1]]]),
            id="#: (2 2 $ 3 _6 0 9)",
        ),
        pytest.param(
            np.array([[[4.2, -1.1]], [[4.2, -1.1]]]),
            np.array([[[[1, 0, 0.2], [1, 1, 0.9]]], [[[1, 0, 0.2], [1, 1, 0.9]]]]),
            id="#: (2 1 2 $ 4.2, _1.1)",
        ),
    ],
)
def test_numberco_monad(y, expected):
    result = numberco_monad(y)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        pytest.param(np.array(1), np.array(2), np.array([[1], [2]]), id="1 ,: 2"),
        pytest.param(
            np.array([1, 2]), np.array(3), np.array([[1, 2], [3, 3]]), id="1 2 ,: 3"
        ),
        pytest.param(
            np.array([1, 2]),
            np.array([3, 4]),
            np.array([[1, 2], [3, 4]]),
            id="1 2 ,: 3 4",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            np.array([5, 6]),
            np.array([[[1, 2], [3, 4]], [[5, 6], [0, 0]]]),
            id="(i.2 2) ,: (5 6)",
        ),
        pytest.param(
            np.array([[0, 1, 2], [3, 4, 5]]),
            np.array([[0, 1], [2, 3], [4, 5]]),
            np.array(
                [[[0, 1, 2], [3, 4, 5], [0, 0, 0]], [[0, 1, 0], [2, 3, 0], [4, 5, 0]]]
            ),
            id="(i. 2 3) ,: (i. 3 2)",
        ),
    ],
)
def test_commaco_dyad(x, y, expected):
    result = commaco_dyad(x, y)
    np.testing.assert_array_equal(result, expected, strict=True)
