"""Methods for applying verb implementations to nouns and verbs.

Main references:
* https://code.jsoftware.com/wiki/Vocabulary/Agreement
* https://www.jsoftware.com/help/jforc/loopless_code_i_verbs_have_r.htm

"""

import functools
from dataclasses import dataclass

import numpy as np
from jinx.errors import JinxNotImplementedError, LengthError, ValenceError
from jinx.execution.numpy.conversion import ndarray_or_scalar_to_noun
from jinx.execution.numpy.helpers import (
    is_ufunc,
    is_ufunc_based,
    maybe_pad_with_fill_value,
)
from jinx.vocabulary import Adverb, Conjunction, Dyad, Monad, Noun, RankT, Verb


def get_rank(verb_rank: RankT, noun_rank: int) -> int:
    """Get the rank at which to apply the verb to the noun.

    If the verb rank is negative, it means that the verb rank is subtracted
    from the noun rank, to a minimum of 0.
    """
    if verb_rank < 0:
        return max(0, noun_rank + verb_rank)  # type: ignore[return-value]
    return min(verb_rank, noun_rank)  # type: ignore[return-value]


def fill_and_assemble(
    cells: list[np.ndarray], frame_shape: tuple[int, ...]
) -> np.ndarray:
    if not cells:
        return np.asarray(cells).reshape(frame_shape)

    cells = maybe_pad_with_fill_value(cells)
    return np.asarray(cells).reshape(frame_shape + cells[0].shape)


@dataclass
class ArrayCells:
    cell_shape: tuple[int, ...]
    frame_shape: tuple[int, ...]
    cells: np.ndarray


def split_into_cells(arr: np.ndarray, rank: int) -> ArrayCells:
    """
    Look at the array shape and rank to determine frame and cell shape.

    The trailing `rank` axes define the cell shape and the preceding
    axes define the frame shape. E.g. for rank=2:

      arr.shape = (n0, n1, n2, n3, n4)
                   ----------  ------
                   ^ frame     ^ cell

    If rank=0, the frame shape is the same as the shape and the monad
    applies to each atom of the array.
    """
    if arr.size == 0:
        return ArrayCells(cell_shape=(), frame_shape=arr.shape, cells=arr)

    if rank == 0:
        return ArrayCells(cell_shape=(), frame_shape=arr.shape, cells=arr.ravel())

    return ArrayCells(
        cell_shape=arr.shape[-rank:],
        frame_shape=arr.shape[:-rank],
        cells=arr.reshape(-1, *arr.shape[-rank:]),
    )


def apply_monad(verb: Verb[np.ndarray], noun: Noun[np.ndarray]) -> Noun[np.ndarray]:
    result = _apply_monad(verb, noun.implementation)
    return ndarray_or_scalar_to_noun(result)


def _apply_monad(verb: Verb[np.ndarray], arr: np.ndarray) -> np.ndarray:
    if verb.monad is None or verb.monad.function is None:
        raise ValenceError(f"Verb {verb.spelling} has no monadic valence.")
    if verb.monad.function is NotImplemented:
        raise JinxNotImplementedError(
            f"Verb {verb.spelling} monad function is not yet implemented in Jinx."
        )

    if isinstance(verb.monad.function, Verb):
        function = functools.partial(_apply_monad, verb.monad.function)
    else:
        function = verb.monad.function  # type: ignore[assignment]

    if arr.size == 0:
        return function(arr)

    rank = get_rank(verb.monad.rank, arr.ndim)

    # If the verb rank is 0 it applies to each atom of the array.
    # NumPy's unary ufuncs are typically designed to work this way
    # Apply the function directly here as an optimisation.
    if rank == 0 and (is_ufunc(function) or is_ufunc_based(function)):
        return function(arr)

    array_cells = split_into_cells(arr, rank)
    cells = [function(cell) for cell in array_cells.cells]
    return fill_and_assemble(cells, array_cells.frame_shape)


def apply_dyad(
    verb: Verb[np.ndarray], noun_1: Noun[np.ndarray], noun_2: Noun[np.ndarray]
) -> Noun[np.ndarray]:
    result = _apply_dyad(verb, noun_1.implementation, noun_2.implementation)
    return ndarray_or_scalar_to_noun(result)


def _apply_dyad(
    verb: Verb[np.ndarray], left_arr: np.ndarray, right_arr: np.ndarray
) -> np.ndarray:
    if verb.dyad is None or verb.dyad.function is None:
        raise ValenceError(f"Verb {verb.spelling} has no dyadic valence.")
    if verb.dyad.function is NotImplemented:
        raise JinxNotImplementedError(
            f"Verb {verb.spelling} dyad function is not yet implemented."
        )

    if isinstance(verb.dyad.function, Verb):
        function = functools.partial(_apply_dyad, verb.dyad.function)
    else:
        function = verb.dyad.function  # type: ignore[assignment]

    if left_arr.size == 0 or right_arr.size == 0:
        return function(left_arr, right_arr)

    left_rank = get_rank(verb.dyad.left_rank, left_arr.ndim)
    right_rank = get_rank(verb.dyad.right_rank, right_arr.ndim)

    # If the left and right ranks are both 0 and one of the arrays is a scalar,
    # apply the dyad directly as an optimisation.
    if (
        left_rank == right_rank == 0
        and (is_ufunc(function) or is_ufunc_based(function))
        and (left_arr.ndim == 0 or right_arr.ndim == 0)
    ):
        return function(left_arr, right_arr)

    left = split_into_cells(left_arr, left_rank)
    right = split_into_cells(right_arr, right_rank)

    # If the left and right frame shapes are the same, we can apply the dyad immediately.
    if left.frame_shape == right.frame_shape:
        cells = [
            function(left_cell, right_cell)
            for left_cell, right_cell in zip(left.cells, right.cells, strict=True)
        ]
        return fill_and_assemble(cells, left.frame_shape)

    # Otherwise we need to find the common frame shape. One of the frame shapes must
    # be a prefix of the other, otherwise it's not possible to apply the dyad.
    common_frame_shape = find_common_frame_shape(left.frame_shape, right.frame_shape)
    if common_frame_shape is None:
        raise LengthError(
            f"Cannot apply dyad {verb.spelling} to arrays of shape {left.frame_shape} and {right.frame_shape}"
        )

    rcf = len(common_frame_shape)

    left_rcf_cell_shape = left_arr.shape[rcf:]
    right_rcf_cell_shape = right_arr.shape[rcf:]

    left_arr_reshaped = left_arr.reshape(-1, *left_rcf_cell_shape)
    right_arr_reshaped = right_arr.reshape(-1, *right_rcf_cell_shape)

    cells = []
    for left_cell, right_cell in zip(
        left_arr_reshaped, right_arr_reshaped, strict=True
    ):
        subcells = []
        if common_frame_shape == left.frame_shape:
            # right_cell is longer and contains multiple operand cells
            if right_rank == 0:
                right_subcells = right_cell.ravel()
            else:
                right_subcells = right_cell.reshape(-1, *right.cell_shape)

            for right_subcell in right_subcells:
                subcells.append(function(left_cell, right_subcell))
        else:
            # left_cell is longer and contains multiple operand cells
            if left_rank == 0:
                left_subcells = left_cell.ravel()
            else:
                left_subcells = left_cell.reshape(-1, *left.cell_shape)

            for left_subcell in left_subcells:
                subcells.append(function(left_subcell, right_cell))

        subcells = maybe_pad_with_fill_value(subcells)
        subarray = np.asarray(subcells)
        if subarray.shape:
            cells.extend(subcells)
        else:
            cells.append(subarray)

    cells = maybe_pad_with_fill_value(cells)
    array = np.asarray(cells)

    # Gather the cells into the final frame shape (the longer of the left
    # and right frame shapes, plus the result cell shape).
    collecting_frame = max(left.frame_shape, right.frame_shape, key=len)
    return array.reshape(collecting_frame + cells[0].shape)


def find_common_frame_shape(
    left_frame_shape: tuple[int, ...], right_frame_shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    if len(left_frame_shape) <= len(right_frame_shape):
        shorter = left_frame_shape
        longer = right_frame_shape
    else:
        shorter = right_frame_shape
        longer = left_frame_shape

    if all(a == b for a, b in zip(shorter, longer)):
        return shorter

    return None


def apply_conjunction(
    verb_or_noun_1: Verb | Noun, conjunction: Conjunction, verb_or_noun_2: Verb | Noun
) -> Verb | Noun:
    return conjunction.function(verb_or_noun_1, verb_or_noun_2)


def apply_adverb(verb_or_noun: Verb | Noun, adverb: Adverb) -> Verb:
    return adverb.function(verb_or_noun)


INFINITY = float("inf")


def build_hook(f: Verb[np.ndarray], g: Verb[np.ndarray]) -> Verb[np.ndarray]:
    """Build a hook given verbs f and g.

      (f g) y  ->  y f (g y)
    x (f g) y  ->  x f (g y)

    The new verb has infinite rank.
    """

    def _monad(y: np.ndarray) -> np.ndarray:
        a = _apply_monad(g, y)
        return _apply_dyad(f, y, a)

    def _dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = _apply_monad(g, y)
        return _apply_dyad(f, x, a)

    f_spelling = f"({f.spelling})" if " " in f.spelling else f.spelling
    g_spelling = f"({g.spelling})" if " " in g.spelling else g.spelling
    spelling = f"{f_spelling} {g_spelling}"

    return Verb[np.ndarray](
        spelling=spelling,
        name=spelling,
        monad=Monad(
            name=spelling,
            rank=INFINITY,
            function=_monad,
        ),
        dyad=Dyad(
            name=spelling,
            left_rank=INFINITY,
            right_rank=INFINITY,
            function=_dyad,
        ),
    )


def build_fork(
    f: Verb[np.ndarray] | Noun[np.ndarray], g: Verb[np.ndarray], h: Verb[np.ndarray]
) -> Verb[np.ndarray]:
    """Build a fork given verbs f, g, h.

      (f g h) y  ->    (f y) g   (h y)
    x (f g h) y  ->  (x f y) g (x h y)

    The new verb has infinite rank.

    Note that f can be a noun, in which case there is one fewer function calls.
    """

    def _monad(y: np.ndarray) -> np.ndarray:
        if isinstance(f, Verb) and f.spelling == "[:":
            hy = _apply_monad(h, y)
            return _apply_monad(g, hy)

        if isinstance(f, Verb):
            a = _apply_monad(f, y)
        else:
            a = f.implementation
        b = _apply_monad(h, y)
        return _apply_dyad(g, a, b)

    def _dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if isinstance(f, Verb) and f.spelling == "[:":
            hy = _apply_dyad(h, x, y)
            return _apply_monad(g, hy)

        if isinstance(f, Verb):
            a = _apply_dyad(f, x, y)
        else:
            a = f.implementation
        b = _apply_dyad(h, x, y)
        return _apply_dyad(g, a, b)

    if isinstance(f, Verb):
        f_spelling = f"({f.spelling})" if " " in f.spelling else f.spelling
    else:
        f_spelling = str(f.implementation)

    g_spelling = f"({g.spelling})" if " " in g.spelling else g.spelling
    h_spelling = f"({h.spelling})" if " " in h.spelling else h.spelling
    spelling = f"{f_spelling} {g_spelling} {h_spelling}"

    return Verb[np.ndarray](
        spelling=spelling,
        name=spelling,
        monad=Monad(
            name=spelling,
            rank=INFINITY,
            function=_monad,
        ),
        dyad=Dyad(
            name=spelling,
            left_rank=INFINITY,
            right_rank=INFINITY,
            function=_dyad,
        ),
    )
