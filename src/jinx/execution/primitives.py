"""Methods implementing J primitives."""

import dataclasses
import operator
from typing import Callable

import numpy as np
import numba

from jinx.vocabulary import Verb, Atom
from jinx.execution.conversion import ensure_noun_implementation, is_ufunc


@numba.vectorize(["float64(int64)", "float64(float64)"], nopython=True)
def percent_monad(y: np.ndarray) -> np.ndarray:
    """% monad: returns the reciprocal of the array."""
    # N.B. np.reciprocal does not support integer types, use division instead.
    return np.divide(1, y)


@numba.vectorize(["int64(int64)", "float64(float64)"], nopython=True)
def ltco_monad(y: np.ndarray) -> np.ndarray:
    """<: monad: decrements the array."""
    return y - 1


@numba.vectorize(["int64(int64)", "float64(float64)"], nopython=True)
def gtco_monad(y: np.ndarray) -> np.ndarray:
    """>: monad: increments the array."""
    return y + 1


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

    # TODO: generate a ufunc on the fly.
    raise NotImplementedError(
        "Adverb '/' only supports commutative operations with ufuncs for now"
    )


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
    "MINUS": (operator.neg, np.subtract),
    "PLUS": (np.conj, np.add),
    "STAR": (np.sign, np.multiply),
    "PERCENT": (percent_monad, np.divide),
    "HAT": (np.exp, np.power),
    "DOLLAR": (dollar_monad, dollar_dyad),
    "LTDOT": (np.floor, np.minimum),
    "LTCO": (ltco_monad, np.less_equal),
    "GTDOT": (np.ceil, np.maximum),
    "GTCO": (gtco_monad, np.greater_equal),
    "IDOT": (idot_monad, None),
    "SLASH": (slash_monad, None),
    "TILDEDOT": (tildedot_monad, None),
    "RANK": rank_conjunction,
}
