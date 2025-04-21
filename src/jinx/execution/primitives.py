"""Methods implementing J primitives.

Where possible, dyads are implemented as ufuncs (either NumPy ufuncs, or
using the numba.vectorize decorator). This equips the dyads with efficient
reduce and accumulate methods over arrays.
"""

import dataclasses
import functools
import itertools
from typing import Callable

import numpy as np
import numba

from jinx.vocabulary import Verb, Atom
from jinx.execution.conversion import ensure_noun_implementation, is_ufunc


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


@numba.vectorize(["int64(int64, int64)"], nopython=True)
def plusco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """+: dyad: not-or operation."""
    # N.B. This is not the same as the J implementation which forbids values
    # outside of 0 and 1.
    return ~np.logical_or(x, y)


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

    if x.size == 1:
        x = np.full_like(y[:1], x[0])
    elif y.size == 1:
        y = np.full_like(x[:1], y[0])
    else:
        trailing_dims = [
            max(xs, ys)
            for xs, ys in itertools.zip_longest(
                reversed(x.shape[1:]), reversed(y.shape[1:]), fillvalue=1
            )
        ]
        trailing_dims.reverse()

        ndmin = max(x.ndim, y.ndim)
        x = increase_ndim(x, ndmin)
        y = increase_ndim(y, ndmin)

        x = np.pad(
            x, [(0, 0)] + [(0, d - s) for s, d in zip(x.shape[1:], trailing_dims)]
        )
        y = np.pad(
            y, [(0, 0)] + [(0, d - s) for s, d in zip(y.shape[1:], trailing_dims)]
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
        raise ValueError(f"Invalid shape: {x}")

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


def slash_monad(verb: Verb) -> Callable[[np.ndarray], np.ndarray]:
    if not verb.dyad:
        raise ValueError(f"Verb {verb.spelling} has no dyadic valence.")

    if verb.dyad.function is None:
        dyad = PRIMITIVE_MAP[verb.name][1]
    else:
        dyad = verb.dyad.function

    if is_ufunc(dyad) and verb.dyad.is_commutative:
        return dyad.reduce

    if is_ufunc(dyad):
        # Not commutative, but dyad has a reduce method.
        # By swapping the arguments and applying it to the
        # reversed array, we can get the same result.
        @numba.vectorize(
            ["int64(int64, int64)", "float64(float64, float64)"], nopython=True
        )
        def _dyad_arg_swap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return dyad(y, x)

        def _dyad_reduce(y: np.ndarray) -> np.ndarray:
            y = np.atleast_1d(y)
            y = np.flip(y, axis=0)
            return _dyad_arg_swap.reduce(y)

        return _dyad_reduce

    # Slow path: dyad is not a ufunc.
    # TODO: Try to find a way to get Numba to compile some examples
    # such as for hooks, where the verbs in the hook are both ufuncs.

    def _dyad_arg_swap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return dyad(y, x)

    def _slow_reduce(y: np.ndarray) -> np.ndarray:
        y = np.atleast_1d(y)
        y = np.flip(y, axis=0)
        return functools.reduce(_dyad_arg_swap, y)

    return _slow_reduce


def rank_conjunction(verb: Verb, noun: Atom) -> Verb:
    ensure_noun_implementation(noun)
    rank = noun.implementation
    spelling = f'{verb.spelling}"{rank}'
    return dataclasses.replace(
        verb,
        spelling=spelling,
        name=spelling,
        monad=dataclasses.replace(verb.monad, rank=rank),
    )


PRIMITIVE_MAP = {
    # NAME: (MONAD, DYAD)
    "EQ": (None, np.equal),
    "MINUS": (np.negative, np.subtract),
    "PLUS": (np.conj, np.add),
    "PLUSCO": (plusco_monad, plusco_dyad),
    "STAR": (np.sign, np.multiply),
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
    "SLASH": (slash_monad, None),
    "TILDEDOT": (tildedot_monad, None),
    "COMMA": (comma_monad, comma_dyad),
    "BAR": (np.abs, bar_dyad),
    "BARDOT": (np.flip, bardot_dyad),
    "RANK": rank_conjunction,
}
