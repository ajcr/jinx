"""Methods implementing J primitives.

Where possible, dyads are implemented as ufuncs. This equips the dyads with
efficient reduce, outer and accumulate methods over arrays.

Specifically where a dyadic application of a verb has left and right rank both 0,
this is equivalent to elementwise application of the verb to the arrays. This is
what ufuncs capture. For example, dyadic `+` is equivalent to `np.add` and dyadic
`*` is equivalent to `np.multiply`.

It is important that all implementations here share the same "rank" characteristics
as their J counterparts.
"""

import dataclasses
import functools
import itertools
import math

import numpy as np

from jinx.vocabulary import Verb, Atom, Array, Monad, Dyad
from jinx.errors import DomainError, ValenceError, JIndexError, LengthError
from jinx.execution.application import _apply_dyad, _apply_monad
from jinx.execution.conversion import is_ufunc, box_dtype, is_box
from jinx.execution.helpers import (
    maybe_pad_with_fill_value,
    maybe_parenthesise_verb_spelling,
)


np.seterr(divide="ignore")


def eq_monad(y: np.ndarray) -> np.ndarray:
    nub = tildedot_monad(y)
    result = []
    for item in nub:
        value = np.all(item == y, axis=tuple(range(1, y.ndim)))
        result.append(value)
    return np.asarray(result).astype(np.int64)


def percent_monad(y: np.ndarray) -> np.ndarray:
    """% monad: returns the reciprocal of the array."""
    # N.B. np.reciprocal does not support integer types, use division instead.
    return 1 / y


def percentco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.power(y, 1 / x)


def plusdot_monad(y: np.ndarray) -> np.ndarray:
    """+. monad: returns real and imaginary parts of numbers."""
    y = np.atleast_1d(y)
    return np.concatenate([np.real(y), np.imag(y)], axis=-1)


def plusco_monad(y: np.ndarray) -> np.ndarray:
    """+: monad: double the values in the array."""
    return 2 * y


def minusdot_monad(y: np.ndarray) -> np.ndarray:
    """-.: monad: returns 1 - y."""
    return 1 - y


def minusco_monad(y: np.ndarray) -> np.ndarray:
    """-: monad: halve the values in the array."""
    return y / 2


def minusco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """-: dyad: match, returns true if x and y have same shape and values."""
    is_equal = np.array_equal(x, y, equal_nan=True)
    return np.asarray(is_equal)


def plusco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """+: dyad: not-or operation."""
    # N.B. This is not the same as the J implementation which forbids values
    # outside of 0 and 1.
    return ~np.logical_or(x, y).astype(np.int64)


def stardot_monad(y: np.ndarray) -> np.ndarray:
    """*. monad: convert x-y coordinates to r-theta coordinates."""
    y = np.atleast_1d(y)
    r = np.abs(y)
    theta = np.angle(y)
    return np.concatenate([r, theta])


def starco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """*: dyad: not-and operation."""
    # N.B. This is not the same as the J implementation which forbids values
    # outside of 0 and 1.
    return ~np.logical_and(x, y).astype(np.int64)


def hatdot_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """^. dyad: logarithm of y to the base x."""
    return np.log(y) / np.log(x)


def lt_monad(y: np.ndarray) -> np.ndarray:
    """< monad: box a noun."""
    return np.array([(y,)], dtype=box_dtype).squeeze()


def gt_monad(y: np.ndarray) -> np.ndarray:
    """< monad: open a boxed element or array of boxed elements."""
    if not is_box(y):
        return y
    elements = [np.asarray(item[0]) for item in y.ravel().tolist()]
    elements_padded = maybe_pad_with_fill_value(elements, fill_value=0)
    return np.asarray(elements_padded).squeeze()


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


def comma_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """, dyad: returns array containing the items of x followed by the items of y."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if x.shape == (1,):
        x = np.full_like(y[:1], x[0])
    elif y.shape == (1,):
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
            [(0, 0)] + [(0, d - s) for s, d in zip(x.shape[1:], trailing_dims)],
        )
        y = np.pad(
            y,
            [(0, 0)] + [(0, d - s) for s, d in zip(y.shape[1:], trailing_dims)],
        )

    # FIX: concatenate loses dtype metadata
    return np.concatenate([x, y], axis=0)


def commadot_monad(y: np.ndarray) -> np.ndarray:
    """,. monad: ravel items."""
    y = np.atleast_1d(y)
    return y.reshape(y.shape[0], -1)


def commadot_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """,. dyad: join each item of x to each item of y."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if x.ndim > 1 and y.ndim > 1 and len(x) != len(y):
        raise LengthError(
            f"Shapes {x.shape} and {y.shape} have different numbers of items"
        )

    if x.shape == (1,):
        x = np.repeat(x, y.shape[0], axis=0)

    if y.shape == (1,):
        y = np.repeat(y, x.shape[0], axis=0)

    result = []
    for x_item, y_item in zip(x, y, strict=True):
        result.append(comma_dyad(x_item, y_item))

    result = maybe_pad_with_fill_value(result, fill_value=0)
    return np.asarray(result)


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


def barco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|: dyad: rearrange the axes of the array."""
    x = np.atleast_1d(x)
    if len(x) > y.ndim:
        raise JIndexError("|: x has more items than y has dimensions")
    if any(item > y.ndim for item in x):
        raise JIndexError("|: x has items greater than y has dimensions")
    if len(set(x)) != len(x):
        raise JIndexError("|: x contains a duplicate axis number")
    first = []
    for i in range(y.ndim):
        if i not in x:
            first.append(i)
    return np.transpose(y, axes=first + x.tolist())


def increase_ndim(y: np.ndarray, ndim: int) -> np.ndarray:
    idx = (np.newaxis,) * (ndim - y.ndim) + (slice(None),)
    return y[idx]


def tildedot_monad(y: np.ndarray) -> np.ndarray:
    """~. monad: remove duplicates from a list."""
    y = np.atleast_1d(y)
    uniq, idx = np.unique(y, return_index=True, axis=0)
    return uniq[np.argsort(idx)]


def tildeco_monad(y: np.ndarray) -> np.ndarray:
    """~: monad: nub sieve."""
    y = np.atleast_1d(y)
    _, idx = np.unique(y, return_index=True, axis=0)
    result = np.zeros(y.shape[0], dtype=np.int64)
    result[idx] = 1
    return result


def dollar_monad(y: np.ndarray) -> np.ndarray | None:
    """$ monad: returns the shape of the array."""
    if np.isscalar(y) or y.shape == ():
        # Differs from the J implementation which returns a missing value for shape of scalar.
        return np.array(0)
    return np.array(y.shape)


def dollar_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """$ dyad: create an array with a particular shape.

    Does not support custom fill values at the moment.
    Does not support INFINITY as an atom of x.
    """
    if np.isscalar(x) and (not np.issubdtype(type(x), np.integer) or x < 0):
        raise DomainError(f"Invalid shape: {x}")

    if np.isscalar(x) or x.shape == ():
        x_shape = (np.squeeze(x),)
    else:
        x_shape = tuple(x)

    if np.isscalar(y) or y.shape == ():
        if is_box(y):
            result = np.array([y] * np.prod(x_shape), dtype=box_dtype).reshape(x_shape)
        else:
            result = np.empty(x_shape, dtype=x.dtype)
            result[:] = y
        return result

    output_shape = x_shape + y.shape[1:]
    data = y.ravel()
    repeat, fill = divmod(np.prod(output_shape), data.size)
    result = np.concatenate([np.tile(data, repeat), data[:fill]]).reshape(output_shape)
    return result


def idot_monad(y: np.ndarray) -> np.ndarray:
    """i. monad: returns increasing/decreasing sequence of integer wrapperd to shape y."""
    arr = np.atleast_1d(y)
    if not np.issubdtype(y.dtype, np.integer):
        raise DomainError(f"y has nonintegral value")
    shape = abs(arr)
    n = np.prod(shape)
    axes_to_flip = np.where(arr < 0)[0]
    result = np.arange(n).reshape(shape)
    return np.flip(result, axes_to_flip)


def number_monad(y: np.ndarray) -> np.ndarray:
    """# monad: count number of items in y."""
    if np.isscalar(y) or y.shape == ():
        return np.array(1)
    return np.array(y.shape[0])


def number_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """# monad: copy items in y exactly x times."""
    return np.repeat(y, x, axis=0)


def squarelf_monad(y: np.ndarray) -> np.ndarray:
    """[ monad: returns the whole array."""
    return y


def squarelf_dyad(x: np.ndarray, _: np.ndarray) -> np.ndarray:
    """[ dyad: returns x."""
    return x


squarerf_monad = squarelf_monad


def squarerf_dyad(_: np.ndarray, y: np.ndarray) -> np.ndarray:
    """] dyad: returns y."""
    return y


def slashco_monad(y: np.ndarray) -> np.ndarray:
    """/: monad: permutation that sorts y in increasing order."""
    y = np.atleast_1d(y)
    if y.ndim == 1:
        return np.argsort(y, stable=True)

    # Ravelled items of y are sorted lexicographically.
    y = y.reshape(len(y), -1)
    return np.lexsort(np.rot90(y))


def slashco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """/: monad: sort y in increasing order."""
    if not np.issubdtype(x.dtype, np.integer):
        raise JIndexError

    y = np.atleast_1d(y)

    if x is y:
        # This handles /:~
        if x.ndim == 1:
            return np.sort(y, kind="stable")
        idx = slashco_monad(y)
        return y[idx]

    # Need to implement '(/: y) { x'
    raise NotImplementedError()


def bslashco_monad(y: np.ndarray) -> np.ndarray:
    r"""\: monad: permutation that sorts y in decreasing order."""
    y = np.atleast_1d(y)
    if y.ndim == 1:
        # Stable sort in decreasing order.
        # np.argsort(a)[::-1] on its own does not work as the indices of
        # equal elements will appear reversed in the result.
        return len(y) - 1 - np.argsort(y[::-1], kind="stable")[::-1]

    y = y.reshape(len(y), -1)
    return len(y) - 1 - np.lexsort(np.rot90(y[::-1]))[::-1]


def bslashco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""\: dyad: sort y in decreasing order."""
    if not np.issubdtype(x.dtype, np.integer):
        raise JIndexError

    y = np.atleast_1d(y)

    if x is y:
        # This handles \:~
        if x.ndim == 1:
            # Not technically correct (see comment on monad above), but
            # good enough for now.
            return np.flip(np.sort(y, kind="stable"))
        idx = bslashco_monad(y)
        return y[idx]

    # Need to implement '(\: y) { x'
    raise NotImplementedError()


def curlylfdot_monad(y: np.ndarray) -> np.ndarray:
    """{. monad: returns the first item of y."""
    y = np.atleast_1d(y)
    if y.size == 0:
        return np.array([])
    return y[0]


def curlylfco_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """{. dyad: the leading x items of y."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if len(x) > y.ndim:
        raise LengthError(f"x has {len(x)} atoms but y has only {y.ndim} axes")

    padding = []
    slices = []

    for dim, take in enumerate(x):
        if take == 0:
            raise NotImplementedError("Dimension with 0 items is not supported")
        if take > y.shape[dim]:
            padding.append((0, take - y.shape[dim]))
            slices.append(slice(None))
        elif take < -y.shape[dim]:
            padding.append((-take - y.shape[dim], 0))
            slices.append(slice(None))
        elif take < 0:
            padding.append((0, 0))
            slices.append(slice(y.shape[dim] + take, None))
        else:
            padding.append((0, 0))
            slices.append(slice(0, take))

    if len(x) < y.ndim:
        padding += [(0, 0)] * (y.ndim - len(x))

    result = y[tuple(slices)]
    result = np.pad(result, padding, mode="constant", constant_values=0)
    return result


def bang_monad(y: np.ndarray) -> np.ndarray:
    """! monad: returns y factorial (and more generally the gamma function of 1+y)."""
    if isinstance(y, int) or np.issubdtype(y.dtype, np.integer) and y >= 0:
        return np.asarray(math.factorial(y))
    return np.asarray(math.gamma(1 + y))


def bang_dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """! dyad: returns y-Combinations-x."""
    if (isinstance(y, int) or np.issubdtype(y.dtype, np.integer) and y >= 0) and (
        isinstance(x, int) or np.issubdtype(x.dtype, np.integer) and x >= 0
    ):
        return np.asarray(math.comb(y, x))
    x_ = bang_monad(x)
    y_ = bang_monad(y)
    x_y = bang_monad(y - x)
    return np.asarray(y_ / x_ / x_y)


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

    else:
        # Slow path: dyad is not a ufunc.
        # The function is either callable, in which cases it is applied directly,
        # or a Verb object that needs to be applied indirectly with _apply_dyad().
        if isinstance(function, Verb):
            func = functools.partial(_apply_dyad, verb)
        else:
            func = function

        def _dyad_arg_swap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return func(y, x)

        def _reduce(y: np.ndarray) -> np.ndarray:
            y = np.atleast_1d(y)
            y = np.flip(y, axis=0)
            return functools.reduce(_dyad_arg_swap, y)

        def _outer(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            verb_slash = _modify_rank(verb, [verb.dyad.left_rank, INFINITY])
            return _apply_dyad(verb_slash, x, y)

        monad = _reduce
        dyad = _outer

    spelling = maybe_parenthesise_verb_spelling(verb.spelling)
    spelling = f"{verb.spelling}/"

    return Verb(
        name=spelling,
        spelling=spelling,
        monad=Monad(name=spelling, rank=INFINITY, function=monad),
        dyad=Dyad(
            name=spelling, left_rank=INFINITY, right_rank=INFINITY, function=dyad
        ),
    )


def bslash_adverb(verb: Verb) -> Verb:
    # Common cases that have a straightforward optimisation.
    SPECIAL_MONAD = {
        "+/": np.add.accumulate,
        "*/": np.multiply.accumulate,
        "<./": np.minimum.accumulate,
        ">./": np.maximum.accumulate,
    }

    if verb.spelling in SPECIAL_MONAD:
        monad_ = SPECIAL_MONAD[verb.spelling]

    else:

        def monad_(y: np.ndarray) -> np.ndarray:
            y = np.atleast_1d(y)
            result = []
            for i in range(1, len(y) + 1):
                result.append(verb.monad.function(y[:i]))
            result = maybe_pad_with_fill_value(result, fill_value=0)
            return np.asarray(result)

    def dyad_(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not np.issubdtype(x.dtype, np.integer):
            raise DomainError(f"x has nonintegral value ({x})")
        y = np.atleast_1d(y)
        if x == 0:
            return np.zeros(len(y) + 1, dtype=np.int64)
        if x == 1 or x == -1:
            windows = y
        elif x > 0:
            # Overlapping windows
            windows = [y[i : i + x] for i in range(len(y) - x + 1)]
        else:
            # Non-overlapping windows
            windows = [y[i : i - x] for i in range(0, len(y), -x)]

        result = []
        for window in windows:
            result.append(verb.monad.function(window))
        result = maybe_pad_with_fill_value(result, fill_value=0)
        return np.asarray(result)

    spelling = maybe_parenthesise_verb_spelling(verb.spelling)
    spelling = f"{spelling}\\"

    return Verb(
        name=spelling,
        spelling=spelling,
        monad=Monad(name=spelling, rank=INFINITY, function=monad_),
        dyad=Dyad(name=spelling, left_rank=0, right_rank=INFINITY, function=dyad_),
    )


def bslashdot_adverb(verb: Verb) -> Verb:
    def monad_(y: np.ndarray) -> np.ndarray:
        y = np.atleast_1d(y)
        result = []
        for i in range(len(y)):
            result.append(verb.monad.function(y[i:]))
        result = maybe_pad_with_fill_value(result, fill_value=0)
        return np.asarray(result)

    def dyad_(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not np.issubdtype(x.dtype, np.integer):
            raise DomainError(f"x has nonintegral value ({x})")
        y = np.atleast_1d(y)
        if x == 0:
            return y
        elif x > 0:
            # Overlapping windows
            windows = [
                np.concatenate([y[:i], y[i + x :]], axis=0)
                for i in range(len(y) - x + 1)
            ]
        else:
            # Non-overlapping windows
            windows = [
                np.concatenate([y[:i], y[i - x :]], axis=0)
                for i in range(0, len(y), -x)
            ]

        result = []
        for window in windows:
            result.append(verb.monad.function(window))
        result = maybe_pad_with_fill_value(result, fill_value=0)
        return np.asarray(result)

    spelling = maybe_parenthesise_verb_spelling(verb.spelling)
    spelling = f"{verb.spelling}\\."

    return Verb(
        name=spelling,
        spelling=spelling,
        monad=Monad(name=spelling, rank=INFINITY, function=monad_),
        dyad=Dyad(name=spelling, left_rank=0, right_rank=INFINITY, function=dyad_),
    )


def tilde_adverb(verb: Verb) -> Verb:
    if verb.dyad.function is None:
        # Note: this differs from J which still allows the adverb to be applied
        # to a verb, but may raise an error when the new verb is applied to a noun
        # and the verb has no dyadic valence.
        raise ValenceError(f"Verb {verb.spelling} has no dyadic valence.")

    def monad(y: np.ndarray) -> np.ndarray:
        # replicate argument and apply verb dyadically
        return _apply_dyad(verb, y, y)

    def dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # swap the arguments and apply verb dyadically
        return _apply_dyad(verb, y, x)

    spelling = maybe_parenthesise_verb_spelling(verb.spelling)
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


def _modify_rank(verb: Verb, rank: np.ndarray | int | float) -> Verb:
    rank = np.atleast_1d(rank)
    if np.issubdtype(rank.dtype, np.floating):
        if not np.isinf(rank).any():
            raise DomainError(f"Rank must be an integer or infinity, got {rank.dtype}")

    elif not np.issubdtype(rank.dtype, np.integer):
        raise DomainError(f"Rank must be an integer or infinity, got {rank.dtype}")

    if rank.size > 3 or rank.ndim > 1:
        raise DomainError(
            f"Rank must be a scalar or 1D array of length <= 3, got {rank.ndim}D array with shape {rank.shape}"
        )

    rank_list = [int(r) if not np.isinf(r) else INFINITY for r in rank.tolist()]
    verb_spelling = spelling = maybe_parenthesise_verb_spelling(verb.spelling)

    if len(rank_list) == 1:
        monad_rank = left_rank = right_rank = rank_list[0]
        spelling = f'{verb_spelling}"{rank_list[0]}'

    elif len(rank_list) == 2:
        left_rank, right_rank = rank_list
        monad_rank = right_rank
        spelling = f'{verb_spelling}"{left_rank} {right_rank}'

    else:
        monad_rank, left_rank, right_rank = rank_list
        spelling = f'{verb_spelling}"{monad_rank} {left_rank} {right_rank}'

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


def rank_conjunction(verb: Verb, noun: Atom | Array) -> Verb:
    rank = np.atleast_1d(noun.implementation).tolist()
    return _modify_rank(verb, rank)


def at_conjunction(u: Verb, v: Verb) -> Verb:
    """@ conjunction: compose verbs u and v, with u applied using the rank of v."""

    # The verb u is to be applied using the rank of v.
    u_rank_v = _modify_rank(u, v.monad.rank)

    def _monad(y: np.ndarray) -> np.ndarray:
        a = _apply_monad(v, y)
        b = _apply_monad(u_rank_v, a)
        return b

    def _dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = _apply_dyad(v, x, y)
        b = _apply_monad(u_rank_v, a)
        return b

    u_spelling = maybe_parenthesise_verb_spelling(u.spelling)
    v_spelling = maybe_parenthesise_verb_spelling(v.spelling)

    if v.dyad is None:
        dyad = None
    else:
        dyad = Dyad(
            name=f"{u_spelling}@{v_spelling}",
            left_rank=v.dyad.left_rank,
            right_rank=v.dyad.right_rank,
            function=_dyad,
        )

    return Verb(
        name=f"{u_spelling}@{v_spelling}",
        spelling=f"{u_spelling}@{v_spelling}",
        monad=Monad(
            name=f"{u_spelling}@{v_spelling}",
            rank=v.monad.rank,
            function=_monad,
        ),
        dyad=dyad,
    )


def atco_conjunction(u: Verb, v: Verb) -> Verb:
    """@: conjunction: compose verbs u and v, with the rank of the new verb as infinity."""

    def monad(y: np.ndarray) -> np.ndarray:
        a = _apply_monad(v, y)
        b = _apply_monad(u, a)
        return b

    def dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = _apply_dyad(v, x, y)
        b = _apply_monad(u, a)
        return b

    u_spelling = maybe_parenthesise_verb_spelling(u.spelling)
    v_spelling = maybe_parenthesise_verb_spelling(v.spelling)

    return Verb(
        name=f"{u_spelling}@:{v_spelling}",
        spelling=f"{u_spelling}@:{v_spelling}",
        monad=Monad(
            name=f"{u_spelling}@:{v_spelling}",
            rank=INFINITY,
            function=monad,
        ),
        dyad=Dyad(
            name=f"{u_spelling}@:{v_spelling}",
            left_rank=INFINITY,
            right_rank=INFINITY,
            function=dyad,
        ),
    )


def ampm_conjunction(left: Verb | Atom | Array, right: Verb | Atom | Array) -> Verb:
    """& conjunction: make a monad from a dyad by providing the left or right noun argument,
    or compose two verbs."""
    if isinstance(left, Atom | Array) and isinstance(right, Verb):
        function = functools.partial(right.dyad.function, left.implementation)
        verb_spelling = maybe_parenthesise_verb_spelling(right.spelling)
        spelling = f"{left.implementation}&{verb_spelling}"
        monad = Monad(name=spelling, rank=right.dyad.right_rank, function=function)
        dyad = None

    elif isinstance(left, Verb) and isinstance(right, Atom | Array):
        # functools.partial cannot be used to apply to right argument of ufuncs
        # as they do not accept kwargs, so we need to wrap the function.
        def _wrapper(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return left.dyad.function(x, y)

        function = functools.partial(_wrapper, y=right.implementation)
        verb_spelling = maybe_parenthesise_verb_spelling(left.spelling)
        spelling = f"{verb_spelling}&{right.implementation}"
        monad = Monad(name=spelling, rank=left.dyad.left_rank, function=function)
        dyad = None

    elif isinstance(left, Verb) and isinstance(right, Verb):
        # Compose u&v, with the new verb having the right verb's monadic rank.
        def monad_(y: np.ndarray) -> np.ndarray:
            a = _apply_monad(right, y)
            b = _apply_monad(left, a)
            return b

        def dyad_(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            ry = _apply_monad(right, y)
            rx = _apply_monad(right, x)
            return _apply_dyad(left, rx, ry)

        left_spelling = maybe_parenthesise_verb_spelling(left.spelling)
        right_spelling = maybe_parenthesise_verb_spelling(right.spelling)
        spelling = f"{left_spelling}&{right_spelling}"

        monad = Monad(name=spelling, rank=right.monad.rank, function=monad_)
        dyad = Dyad(
            name=spelling,
            left_rank=right.monad.rank,
            right_rank=right.monad.rank,
            function=dyad_,
        )

    return Verb(name=spelling, spelling=spelling, monad=monad, dyad=dyad)


def ampdotco_conjunction(u: Verb, v: Verb) -> Verb:
    """&.: conjunction: execute v on the arguments, then u on the result, then
    the inverse v of on that result."""

    if v.obverse is None:
        raise DomainError(f"{v.spelling} has no obverse")

    def _monad(y: np.ndarray) -> np.ndarray:
        vy = _apply_monad(v, y)
        uvy = _apply_monad(u, vy)
        return _apply_monad(v.obverse, uvy)

    def _dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        vy = _apply_monad(v, y)
        vx = _apply_monad(v, x)
        uvy = _apply_dyad(u, vx, vy)
        return _apply_monad(v.obverse, uvy)

    v_spelling = maybe_parenthesise_verb_spelling(v.spelling)
    u_spelling = maybe_parenthesise_verb_spelling(u.spelling)

    return Verb(
        name=f"{u_spelling}&.:{v_spelling}",
        spelling=f"{u_spelling}&.:{v_spelling}",
        monad=Monad(
            name=f"{u_spelling}&.:{v_spelling}",
            rank=INFINITY,
            function=_monad,
        ),
        dyad=Dyad(
            name=f"{u_spelling}&.:{v_spelling}",
            left_rank=INFINITY,
            right_rank=INFINITY,
            function=_dyad,
        ),
    )


def hatco_conjunction(u: Verb, noun_or_verb: Atom | Array | Verb) -> Verb:
    """^: conjunction: power of verb."""

    if isinstance(noun_or_verb, Verb):
        raise NotImplementedError("^: conjunction with verb is not yet implemented")

    if isinstance(noun_or_verb, Atom | Array):
        exponent: Atom | Array = noun_or_verb

    if exponent.implementation.size == 0:
        raise DomainError(
            f"^: requires non-empty exponent, got {exponent.implementation}"
        )

    # Special case (^:0) is ]
    if (
        np.isscalar(exponent.implementation) or exponent.implementation.shape == ()
    ) and exponent == 0:
        return Verb(
            name="SQUARELF",
            spelling="]",
            monad=squarelf_monad,
            dyad=squarelf_dyad,
            obverse="]",
        )

    # Special case (^:1) is u
    if (
        np.isscalar(exponent.implementation) or exponent.implementation.shape == ()
    ) and exponent == 1:
        return u

    if np.isinf(exponent.implementation).any():
        raise NotImplementedError(f"^: with infinite exponent is not yet implemented")

    if not np.issubdtype(exponent.implementation.dtype, np.integer):
        raise DomainError(
            f"^: requires integer exponent, got {exponent.implementation}"
        )

    def monad(y: np.ndarray) -> np.ndarray:
        result = []
        for atom in exponent.implementation.ravel().tolist():
            if atom == 0:
                result.append(y)
                continue
            elif atom > 0:
                verb = u
                exp = atom
            else:  # atom < 0:
                if not isinstance(u.obverse, Verb):
                    raise DomainError(f"{u.spelling} has no obverse")
                verb = u.obverse
                exp = -atom

            atom_result = y
            for _ in range(exp):
                atom_result = _apply_monad(verb, atom_result)

            result.append(atom_result)

        result = maybe_pad_with_fill_value(result, fill_value=0)
        result = np.asarray(result)
        return result.reshape(exponent.implementation.shape + result[0].shape)

    def dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        result = []
        for atom in exponent.implementation.ravel().tolist():
            if atom == 0:
                result.append(y)
                continue
            elif atom > 0:
                verb = u
                exp = atom
            else:  # atom < 0:
                if not isinstance(u.obverse, Verb):
                    raise DomainError(f"{u.spelling} has no obverse")
                verb = u.obverse
                exp = -atom

            atom_result = y
            for _ in range(exp):
                atom_result = _apply_dyad(verb, x, atom_result)

            result.append(atom_result)

        result = maybe_pad_with_fill_value(result, fill_value=0)
        result = np.asarray(result)
        return result.reshape(exponent.implementation.shape + result[0].shape)

    u_spelling = maybe_parenthesise_verb_spelling(u.spelling)

    return Verb(
        name=f"{u_spelling}^:{exponent.implementation}",
        spelling=f"{u_spelling}^:{exponent.implementation}",
        monad=Monad(
            name=f"{u_spelling}^:{exponent.implementation}",
            rank=INFINITY,
            function=monad,
        ),
        dyad=Dyad(
            name=f"{u_spelling}^:{exponent.implementation}",
            left_rank=INFINITY,
            right_rank=INFINITY,
            function=dyad,
        ),
    )


# Use NotImplemented for monads or dyads that have not yet been implemented in Jinx.
# Use None for monadic or dyadic valences of the verb do not exist in J.
PRIMITIVE_MAP = {
    # VERB: (MONAD, DYAD)
    "EQ": (eq_monad, np.equal),
    "MINUS": (np.negative, np.subtract),
    "MINUSDOT": (minusdot_monad, NotImplemented),
    "MINUSCO": (minusco_monad, minusco_dyad),
    "PLUS": (np.conj, np.add),
    "PLUSDOT": (plusdot_monad, np.gcd),
    "PLUSCO": (plusco_monad, plusco_dyad),
    "STAR": (np.sign, np.multiply),
    "STARDOT": (stardot_monad, np.lcm),
    "STARCO": (np.square, starco_dyad),
    "PERCENT": (percent_monad, np.divide),
    "PERCENTCO": (np.sqrt, percentco_dyad),
    "HAT": (np.exp, np.power),
    "HATDOT": (np.log, hatdot_dyad),
    "DOLLAR": (dollar_monad, dollar_dyad),
    "LT": (lt_monad, np.less),
    "LTDOT": (np.floor, np.minimum),
    "LTCO": (ltco_monad, np.less_equal),
    "GT": (gt_monad, np.greater),
    "GTDOT": (np.ceil, np.maximum),
    "GTCO": (gtco_monad, np.greater_equal),
    "IDOT": (idot_monad, NotImplemented),
    "TILDEDOT": (tildedot_monad, None),
    "TILDECO": (tildeco_monad, np.not_equal),
    "COMMA": (comma_monad, comma_dyad),
    "COMMADOT": (commadot_monad, commadot_dyad),
    "BAR": (np.abs, bar_dyad),
    "BARDOT": (np.flip, bardot_dyad),
    "BARCO": (np.transpose, barco_dyad),
    "NUMBER": (number_monad, number_dyad),
    "SQUARELF": (squarelf_monad, squarelf_dyad),
    "SQUARERF": (squarerf_monad, squarerf_dyad),
    "SLASHCO": (slashco_monad, slashco_dyad),
    "BSLASHCO": (bslashco_monad, bslashco_dyad),
    "BANG": (bang_monad, bang_dyad),
    "CURLYLFDOT": (curlylfdot_monad, curlylfco_dyad),
    # ADVERB: adverb
    "SLASH": slash_adverb,
    "BSLASH": bslash_adverb,
    "BSLASHDOT": bslashdot_adverb,
    "TILDE": tilde_adverb,
    # CONJUNCTION: conjunction
    "RANK": rank_conjunction,
    "AT": at_conjunction,
    "ATCO": atco_conjunction,
    "AMPM": ampm_conjunction,
    "AMPDOTCO": ampdotco_conjunction,
    "HATCO": hatco_conjunction,
}
