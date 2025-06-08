"""Methods for applying verb implementations to nouns and verbs.

Main references:
* https://code.jsoftware.com/wiki/Vocabulary/Agreement
* https://www.jsoftware.com/help/jforc/loopless_code_i_verbs_have_r.htm

"""

import functools

import numpy as np

from jinx.vocabulary import Noun, Verb, Conjunction, Adverb, Monad, Dyad, Atom, Array
from jinx.errors import LengthError
from jinx.execution.conversion import (
    ndarray_or_scalar_to_noun,
    is_ufunc,
    asarray_boxsafe,
)
from jinx.execution.helpers import maybe_pad_with_fill_value


def apply_monad(verb: Verb, noun: Noun) -> np.ndarray:
    arr = noun.implementation

    result = _apply_monad(verb, arr)
    return ndarray_or_scalar_to_noun(result)


def _apply_monad(verb: Verb, arr: np.ndarray) -> np.ndarray:
    if isinstance(verb.monad.function, Verb):
        function = functools.partial(_apply_monad, verb.monad.function)
    else:
        function = verb.monad.function

    noun_rank = arr.ndim
    verb_rank = verb.monad.rank

    # If the verb rank is negative, it means that the verb rank is subtracted
    # from the operand rank, to a minimum of 0.
    if verb_rank < 0:
        verb_rank = max(0, noun_rank + verb_rank)

    rank = min(verb_rank, noun_rank)

    # If the verb rank is 0 it applies to each atom of the array.
    # NumPy's unary ufuncs are typically designed to work this way
    # Apply the function directly here as an optimisation.
    if rank == 0 and is_ufunc(function):
        return function(arr)

    # Look at the shape of the array and the rank of the verb to
    # determine the frame and cell shape.
    #
    # The trailing `rank` axes define the cell shape and the preceding
    # axes define the frame shape. E.g. for rank=2:
    #
    #   arr.shape = (n0, n1, n2, n3, n4)
    #                ----------  ------
    #                ^ frame     ^ cell
    #
    # If rank=0, the frame shape is the same as the shape and the monad
    # applies to each atom of the array.
    if rank == 0:
        frame_shape = arr.shape
        arr_reshaped = arr.ravel()
    else:
        cell_shape = arr.shape[-rank:]
        frame_shape = arr.shape[:-rank]
        arr_reshaped = arr.reshape(-1, *cell_shape)

    # Apply the function to each cell of the reshaped array, then add any
    # padding needed to make all cells the same shape. Put the cells into
    # the frame and return the result.
    cells = [function(cell) for cell in arr_reshaped]
    cells = maybe_pad_with_fill_value(cells)
    result = asarray_boxsafe(cells).reshape(frame_shape + cells[0].shape)
    return result


def apply_dyad(verb: Verb, noun_1: Noun, noun_2: Noun) -> Noun:
    left_arr = noun_1.implementation
    right_arr = noun_2.implementation

    result = _apply_dyad(verb, left_arr, right_arr)
    return ndarray_or_scalar_to_noun(result)


def _apply_dyad(verb: Verb, left_arr: np.ndarray, right_arr: np.ndarray) -> np.ndarray:
    if isinstance(verb.dyad.function, Verb):
        function = functools.partial(_apply_dyad, verb.dyad.function)
    else:
        function = verb.dyad.function

    left_rank = verb.dyad.left_rank
    right_rank = verb.dyad.right_rank

    if left_rank < 0:
        left_rank = max(0, left_arr.ndim + left_rank)

    if right_rank < 0:
        right_rank = max(0, right_arr.ndim + right_rank)

    left_rank = min(left_arr.ndim, verb.dyad.left_rank)
    right_rank = min(right_arr.ndim, verb.dyad.right_rank)

    # If the left and right ranks are both 0 and one of the arrays is a scalar,
    # apply the dyad directly as an optimisation.
    if (
        left_rank == right_rank == 0
        and is_ufunc(function)
        and (left_arr.ndim == 0 or right_arr.ndim == 0)
    ):
        return function(left_arr, right_arr)

    left_cell_shape = left_arr.shape[-left_rank:] if left_rank else ()
    left_frame_shape = left_arr.shape[:-left_rank] if left_rank else left_arr.shape
    left_arr_reshaped = (
        left_arr.reshape(-1, *left_cell_shape) if left_rank else left_arr.ravel()
    )

    right_cell_shape = right_arr.shape[-right_rank:] if right_rank else ()
    right_frame_shape = right_arr.shape[:-right_rank] if right_rank else right_arr.shape
    right_arr_reshaped = (
        right_arr.reshape(-1, *right_cell_shape) if right_rank else right_arr.ravel()
    )

    # If the left and right frame shapes are the same, we can apply the dyad immediately.
    if left_frame_shape == right_frame_shape:
        cells = [
            function(left_cell, right_cell)
            for left_cell, right_cell in zip(
                left_arr_reshaped, right_arr_reshaped, strict=True
            )
        ]
        cells = maybe_pad_with_fill_value(cells)
        result = np.asarray(cells).reshape(left_frame_shape + cells[0].shape)
        return result

    # Otherwise we need to find the common frame shape. One of the frame shapes must
    # be a prefix of the other, otherwise it's not possible to apply the dyad.
    common_frame_shape = find_common_frame_shape(left_frame_shape, right_frame_shape)
    if common_frame_shape is None:
        raise LengthError(
            f"Cannot apply dyad {verb.spelling} to arrays of shape {left_frame_shape} and {right_frame_shape}"
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
        if common_frame_shape == left_frame_shape:
            # right_cell is longer and contains multiple operand cells
            if right_rank == 0:
                right = right_cell.ravel()
            else:
                right = right_cell.reshape(-1, *right_cell_shape)

            for right_subcell in right:
                subcells.append(function(left_cell, right_subcell))
        else:
            # left_cell is longer and contains multiple operand cells
            if left_rank == 0:
                left = left_cell.ravel()
            else:
                left = left_cell.reshape(-1, *left_cell_shape)

            for left_subcell in left:
                subcells.append(function(left_subcell, right_cell))

        subcells = maybe_pad_with_fill_value(subcells)
        subcells = np.asarray(subcells)
        if subcells.shape:
            cells.extend(subcells)
        else:
            cells.append(subcells)

    cells = maybe_pad_with_fill_value(cells)
    cells = np.asarray(cells)

    # Gather the cells into the final frame shape (the longer of the left
    # and right frame shapes, plus the result cell shape).
    collecting_frame = max(left_frame_shape, right_frame_shape, key=len)
    return cells.reshape(collecting_frame + cells[0].shape)


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


def build_hook(f: Verb, g: Verb) -> Verb:
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

    return Verb(
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


def build_fork(f: Verb | Atom | Array, g: Verb, h: Verb) -> Verb:
    """Build a fork given verbs f, g, h.

      (f g h) y  ->    (f y) g   (h y)
    x (f g h) y  ->  (x f y) g (x h y)

    The new verb has infinite rank.

    Note that f can be a noun, in which case there is one fewer function calls.
    """

    def _monad(y: np.ndarray) -> np.ndarray:
        if isinstance(f, Verb):
            a = _apply_monad(f, y)
        else:
            a = f.implementation
        b = _apply_monad(h, y)
        return _apply_dyad(g, a, b)

    def _dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if isinstance(f, Verb):
            a = _apply_dyad(f, x, y)
        else:
            a = f.implementation
        b = _apply_dyad(h, x, y)
        return _apply_dyad(g, a, b)

    if isinstance(f, Verb):
        f_spelling = f"({f.spelling})" if " " in f.spelling else f.spelling
    else:
        f_spelling = f.implementation
    g_spelling = f"({g.spelling})" if " " in g.spelling else g.spelling
    h_spelling = f"({h.spelling})" if " " in h.spelling else h.spelling

    spelling = f"{f_spelling} {g_spelling} {h_spelling}"

    return Verb(
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
