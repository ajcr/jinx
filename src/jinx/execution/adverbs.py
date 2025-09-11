"""Methods implementing J adverbs."""

import dataclasses
import functools

import numpy as np

from jinx.vocabulary import Verb, Monad, Dyad
from jinx.errors import DomainError, ValenceError, JinxNotImplementedError, LengthError
from jinx.execution.application import _apply_dyad, _apply_monad
from jinx.execution.helpers import (
    is_ufunc,
    is_box,
    get_fill_value,
    maybe_pad_with_fill_value,
    maybe_parenthesise_verb_spelling,
)


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
                result.append(_apply_monad(verb, y[:i]))
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
            result.append(_apply_monad(verb, window))
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
    SPECIAL_MONAD = {
        "+/": lambda x: np.add.accumulate(x[::-1])[::-1],
        "*/": lambda x: np.multiply.accumulate(x[::-1])[::-1],
        "<./": lambda x: np.minimum.accumulate(x[::-1])[::-1],
        ">./": lambda x: np.maximum.accumulate(x[::-1])[::-1],
    }

    if verb.spelling in SPECIAL_MONAD:
        monad_ = SPECIAL_MONAD[verb.spelling]
    else:

        def monad_(y: np.ndarray) -> np.ndarray:
            y = np.atleast_1d(y)
            result = []
            for i in range(len(y)):
                result.append(_apply_monad(verb, y[i:]))
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
            result.append(_apply_monad(verb, window))
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


def slashdot_adverb(verb: Verb) -> Verb:
    def monad(y: np.ndarray) -> np.ndarray:
        y = np.atleast_1d(y)

        if y.ndim == 1:
            result = [_apply_monad(verb, item) for item in y]
        elif y.ndim <= 3:
            result = []
            for offset in range(1 - y.shape[0], y.shape[1]):
                item = np.diagonal(y[::-1], offset).T[::-1]
                result.append(_apply_monad(verb, item))
        else:
            JinxNotImplementedError(
                f"Monad {verb.spelling} dooes not yet support array rank > 3."
            )

        result = maybe_pad_with_fill_value(result, fill_value=0)
        return np.asarray(result)

    def dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if len(x) != len(y):
            raise LengthError(
                f"x and y must have the same length, got {len(x)} and {len(y)}"
            )

        item_indices = {}

        if is_box(x):
            for i, x_item in enumerate(x):
                item_indices.setdefault(x_item[0].tobytes(), []).append(i)

        else:
            for i, x_item in enumerate(x):
                item_indices.setdefault(x_item.tobytes(), []).append(i)

        result = []
        for idx in item_indices.values():
            result.append(_apply_monad(verb, y[idx]))

        result = maybe_pad_with_fill_value(result, fill_value=get_fill_value(y))
        return np.asarray(result)

    spelling = maybe_parenthesise_verb_spelling(verb.spelling)
    spelling = f"{verb.spelling}/."

    return Verb(
        name=spelling,
        spelling=spelling,
        monad=Monad(name=spelling, rank=INFINITY, function=monad),
        dyad=Dyad(
            name=spelling,
            left_rank=INFINITY,
            right_rank=INFINITY,
            function=dyad,
        ),
    )


ADVERB_MAP = {
    "SLASH": slash_adverb,
    "SLASHDOT": slashdot_adverb,
    "BSLASH": bslash_adverb,
    "BSLASHDOT": bslashdot_adverb,
    "TILDE": tilde_adverb,
}
