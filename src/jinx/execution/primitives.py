"""Methods implementing J primitives.

Where possible, dyads are implemented as ufuncs (either NumPy ufuncs, or
using the numba.vectorize decorator). This equips the dyads with efficient
reduce and accumulate methods over arrays.

It is important that the implementations here share the same "rank" characteristics
as their J counterparts. For example the dyadic `+` operator in J has left and right
rank 0, meaning that it operates on the "atoms" of each array (i.e. the numeric value
in the array), not the array or its metadata.
"""

import dataclasses
import functools
import itertools

import numpy as np
import numba

from jinx.vocabulary import Verb, Atom, Array, Monad, Dyad
from jinx.errors import DomainError, ValenceError
from jinx.execution.conversion import is_ufunc


def percent_monad(y: np.ndarray) -> np.ndarray:
    """% monad: returns the reciprocal of the array."""
    # N.B. np.reciprocal does not support integer types, use division instead.
    return 1 / y


@numba.vectorize(["float64(int64, int64)", "float64(float64, float64)"], nopython=True)
def percentco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.power(y, 1 / x)


def plusco_monad(y: np.ndarray) -> np.ndarray:
    """+: monad: double the values in the array."""
    return 2 * y


@numba.vectorize(["float64(int64)", "float64(float64)"], nopython=True)
def minusco_monad(y: np.ndarray) -> np.ndarray:
    """-: monad: halve the values in the array."""
    return y / 2


def minusco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """-: dyad: match, returns true if x and y have same shape and values."""
    is_equal = np.array_equal(x, y, equal_nan=True)
    return np.asarray(is_equal)


@numba.vectorize(["int64(int64, int64)"], nopython=True)
def plusco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """+: dyad: not-or operation."""
    # N.B. This is not the same as the J implementation which forbids values
    # outside of 0 and 1.
    return ~np.logical_or(x, y)


@numba.vectorize(["int64(int64, int64)"], nopython=True)
def starco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """*: dyad: not-and operation."""
    # N.B. This is not the same as the J implementation which forbids values
    # outside of 0 and 1.
    return ~np.logical_and(x, y)


@numba.vectorize(["float64(float64, float64)"], nopython=True)
def hatdot_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """^. dyad: logarithm of y to the base x."""
    return np.log(y) / np.log(x)


def ltco_monad(y: np.ndarray) -> np.ndarray:
    """<: monad: decrements the array."""
    return y - 1


def gtco_monad(y: np.ndarray) -> np.ndarray:
    """>: monad: increments the array."""
    return y + 1


def comma_monad(y: np.ndarray) -> np.ndarray:
    """, monad: returns the flattened array."""
    y = np.atleast_1d(y)
    return np.ravel(y)


@numba.vectorize(["int64(int64, int64)", "float64(float64, float64)"], nopython=True)
def bar_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """| dyad: remainder when dividing y by x."""
    return np.mod(y, x)


def bardot_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|. dyad: rotate the array."""
    y = np.atleast_1d(y)
    x = np.atleast_1d(x)
    if x.shape[-1] > y.ndim:
        raise ValueError(
            f"length error, executing dyad |. (x has {x.shape[-1]} atoms but y only has {y.ndim} axes)"
        )
    return np.roll(y, -x, axis=tuple(range(x.shape[-1])))


def increase_ndim(y: np.ndarray, ndim: int) -> np.ndarray:
    idx = (np.newaxis,) * (ndim - y.ndim) + (slice(None),)
    return y[idx]


def comma_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """, dyad: returns array containing the items of x followed by the items of y."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if x.size == 1 and x.ndim == 1:
        x = np.full_like(y[:1], x[0])
    elif y.size == 1 and y.ndim == 1:
        y = np.full_like(x[:1], y[0])
    else:
        trailing_dims = [
            max(xs, ys)
            for xs, ys in itertools.zip_longest(
                reversed(x.shape), reversed(y.shape), fillvalue=1
            )
        ]
        trailing_dims.reverse()
        trailing_dims = trailing_dims[1:]  # ignore dimension that we concatenate along

        ndmin = max(x.ndim, y.ndim)
        x = increase_ndim(x, ndmin)
        y = increase_ndim(y, ndmin)

        x = np.pad(
            x,
            [(0, 0)] + [(0, max(d - s, 0)) for s, d in zip(x.shape[1:], trailing_dims)],
        )
        y = np.pad(
            y,
            [(0, 0)] + [(0, max(d - s, 0)) for s, d in zip(y.shape[1:], trailing_dims)],
        )

    return np.concatenate([x, y], axis=0)


def dollar_monad(y: np.ndarray) -> np.ndarray | None:
    """$ monad: returns the shape of the array."""
    if np.isscalar(y) or y.size == 1:
        # Differs from the J implementation which returns a missing value for shape of scalar.
        return np.array([0])
    return np.array(y.shape)


def tildedot_monad(y: np.ndarray) -> np.ndarray:
    """~. monad: remove duplicates from a list."""
    y = np.atleast_1d(y)
    uniq, idx = np.unique(y, return_index=True, axis=0)
    return uniq[np.argsort(idx)]


def dollar_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """$ dyad: create an array with a particular shape.

    Does not support custom fill values at the moment.
    Does not support INFINITY as an atom of x.
    """
    if np.isscalar(x) and (not np.issubdtype(type(x), np.integer) or x < 0):
        raise DomainError(f"Invalid shape: {x}")

    if np.isscalar(x) or x.size == 1:
        x_shape = (x,)
    else:
        x_shape = tuple(x)

    if np.isscalar(y) or y.size == 1:
        result = np.zeros(x_shape, dtype=x.dtype)
        result[:] = y
        return result

    output_shape = x_shape + y.shape[1:]
    data = y.ravel()
    repeat, fill = divmod(np.prod(output_shape), data.size)
    result = np.concatenate([np.tile(data, repeat), data[:fill]]).reshape(output_shape)
    return result


def idot_monad(y: np.ndarray) -> np.ndarray:
    arr = np.atleast_1d(y)
    shape = abs(arr)
    n = np.prod(shape)
    axes_to_flip = np.where(arr < 0)[0]
    result = np.arange(n).reshape(shape)
    return np.flip(result, axes_to_flip)


def tally_monad(y: np.ndarray) -> np.ndarray:
    """# monad: count number of items in y."""
    if np.isscalar(y) or y.size == 1 and y.ndim <= 1:
        return 0
    return y.shape[0]


def tally_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """# monad: copy items in y exactly x times."""
    return np.repeat(y, x, axis=0)


INFINITY = float("inf")


def slash_adverb(verb: Verb) -> Verb:
    function = verb.dyad.function
    if function is None:
        # Note: this differs from J which still allows the adverb to be applied
        # to a verb, but may raise an error when the new verb is applied to a noun
        # and the verb has no dyadic valence.
        raise ValenceError(f"Verb {verb.spelling} has no dyadic valence.")

    if is_ufunc(function) and verb.dyad.is_commutative:
        monad = function.reduce
        dyad = function.outer

    elif is_ufunc(function):
        # Not commutative, but dyad has a reduce method.
        # By swapping the arguments and applying it to the
        # reversed array, we can get the same result.
        @numba.vectorize(
            ["int64(int64, int64)", "float64(float64, float64)"], nopython=True
        )
        def _dyad_arg_swap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return function(y, x)

        def _reduce(y: np.ndarray) -> np.ndarray:
            y = np.atleast_1d(y)
            y = np.flip(y, axis=0)
            return _dyad_arg_swap.reduce(y)

        monad = _reduce
        dyad = function.outer

    elif callable(function):
        # Slow path: dyad is not a ufunc.
        # TODO: Try to find a way to get Numba to compile some examples
        # such as for hooks, where the verbs in the hook are both ufuncs.

        def _dyad_arg_swap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return function(y, x)

        def _reduce(y: np.ndarray) -> np.ndarray:
            y = np.atleast_1d(y)
            y = np.flip(y, axis=0)
            return functools.reduce(_dyad_arg_swap, y)

        # This gives incorrect results for some verbs, for example
        # comma: (i.6),/(i.2 2).
        # TODO: fix this.
        def _outer(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            x = np.atleast_1d(x)
            y = np.atleast_1d(y)
            table = []
            for x_item in x:
                row = []
                for y_item in y:
                    value = function(x_item, y_item)
                    row.append(value)
                table.append(row)
            return np.asarray(table)

        monad = _reduce
        dyad = _outer

    else:
        raise NotImplementedError(
            f"Adverb / cannot yet be applied to verb '{verb.spelling}'"
        )

    if " " in verb.spelling:
        spelling = f"({verb.spelling})/"
    else:
        spelling = f"{verb.spelling}/"

    return Verb(
        name=spelling,
        spelling=spelling,
        monad=Monad(name=spelling, rank=INFINITY, function=monad),
        dyad=Dyad(
            name=spelling, left_rank=INFINITY, right_rank=INFINITY, function=dyad
        ),
    )


def tilde_adverb(verb: Verb) -> Verb:
    function = verb.dyad.function
    if function is None:
        # Note: this differs from J which still allows the adverb to be applied
        # to a verb, but may raise an error when the new verb is applied to a noun
        # and the verb has no dyadic valence.
        raise ValenceError(f"Verb {verb.spelling} has no dyadic valence.")

    def monad(y: np.ndarray) -> np.ndarray:
        return function(y, y)

    def dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return function(y, x)

    if " " in verb.spelling:
        spelling = f"({verb.spelling})~"
    else:
        spelling = f"{verb.spelling}~"

    return Verb(
        name=spelling,
        spelling=spelling,
        monad=Monad(name=spelling, rank=INFINITY, function=monad),
        dyad=Dyad(
            name=spelling,
            left_rank=verb.dyad.right_rank,
            right_rank=verb.dyad.left_rank,
            function=dyad,
        ),
    )


def rank_conjunction(verb: Verb, noun: Atom | Array) -> Verb:
    rank = np.atleast_1d(noun.implementation)

    if not np.issubdtype(rank.dtype, np.integer):
        raise DomainError(f"Rank must be an integer, got {rank.dtype}")

    if rank.size > 3 or rank.ndim > 1:
        raise DomainError(
            f"Rank must be a scalar or 1D array of length 2, got {rank.ndim}D array with shape {rank.shape}"
        )

    if rank.size == 1:
        monad_rank = left_rank = right_rank = rank[0]
        spelling = f'{verb.spelling}"{rank[0]}'

    elif rank.size == 2:
        left_rank, right_rank = rank
        monad_rank = right_rank
        spelling = f'{verb.spelling}"{left_rank} {right_rank}'

    else:
        monad_rank, left_rank, right_rank = rank
        spelling = f'{verb.spelling}"{monad_rank} {left_rank} {right_rank}'

    if verb.monad:
        monad = dataclasses.replace(verb.monad, rank=monad_rank, function=verb)
    else:
        monad = None

    if verb.dyad:
        dyad = dataclasses.replace(
            verb.dyad,
            left_rank=left_rank,
            right_rank=right_rank,
            function=verb,
        )
    else:
        dyad = None

    return dataclasses.replace(
        verb,
        spelling=spelling,
        name=spelling,
        monad=monad,
        dyad=dyad,
    )


PRIMITIVE_MAP = {
    # NAME: (MONAD, DYAD)
    "EQ": (None, np.equal),
    "MINUS": (np.negative, np.subtract),
    "MINUSCO": (minusco_monad, minusco_dyad),
    "PLUS": (np.conj, np.add),
    "PLUSCO": (plusco_monad, plusco_dyad),
    "STAR": (np.sign, np.multiply),
    "STARCO": (np.square, starco_dyad),
    "PERCENT": (percent_monad, np.divide),
    "PERCENTCO": (np.sqrt, percentco_dyad),
    "HAT": (np.exp, np.power),
    "HATDOT": (np.log, hatdot_dyad),
    "DOLLAR": (dollar_monad, dollar_dyad),
    "LTDOT": (np.floor, np.minimum),
    "LTCO": (ltco_monad, np.less_equal),
    "GTDOT": (np.ceil, np.maximum),
    "GTCO": (gtco_monad, np.greater_equal),
    "IDOT": (idot_monad, None),
    "SLASH": (slash_adverb, None),
    "TILDE": (tilde_adverb, None),
    "TILDEDOT": (tildedot_monad, None),
    "COMMA": (comma_monad, comma_dyad),
    "BAR": (np.abs, bar_dyad),
    "BARDOT": (np.flip, bardot_dyad),
    "RANK": rank_conjunction,
    "NUMBER": (tally_monad, tally_dyad),
}
