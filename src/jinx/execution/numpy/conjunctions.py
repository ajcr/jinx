"""Methods implementing J conjunctions."""

import dataclasses
import functools

import numpy as np
from jinx.errors import DomainError, JinxNotImplementedError, ValenceError
from jinx.execution.numpy.application import (
    _apply_dyad,
    _apply_monad,
    fill_and_assemble,
    get_rank,
    split_into_cells,
)
from jinx.execution.numpy.helpers import (
    maybe_pad_with_fill_value,
    maybe_parenthesise_verb_spelling,
)
from jinx.vocabulary import Dyad, Monad, Noun, Verb

INFINITY = float("inf")


def _modify_rank(
    verb: Verb[np.ndarray], rank: np.ndarray | int | float
) -> Verb[np.ndarray]:
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


def rank_conjunction(
    verb: Verb[np.ndarray], noun: Noun[np.ndarray]
) -> Verb[np.ndarray]:
    rank = np.atleast_1d(noun.implementation).tolist()
    return _modify_rank(verb, rank)


def at_conjunction(u: Verb[np.ndarray], v: Verb[np.ndarray]) -> Verb[np.ndarray]:
    """@ conjunction: compose verbs u and v, with u applied using the rank of v.

    This means v is applied first, then u is applied to each cell of the result of v
    *before* any padding or assembly is done.

    Once u has been applied to each v-cell, the results are padded and assembled.
    """
    if v.monad is None:
        raise ValenceError(f"{v.spelling} has no monadic form")

    def _monad(y: np.ndarray) -> np.ndarray:
        rank = get_rank(v.monad.rank, y.ndim)
        v_cell_array = split_into_cells(y, rank)
        v_cells = [_apply_monad(v, cell) for cell in v_cell_array.cells]
        u_cells = [_apply_monad(u, cell) for cell in v_cells]
        padded_cells = maybe_pad_with_fill_value(u_cells)
        return fill_and_assemble(padded_cells, v_cell_array.frame_shape)

    def _dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        left_rank = get_rank(v.dyad.left_rank, x.ndim)
        right_rank = get_rank(v.dyad.right_rank, y.ndim)

        v_cell_left_array = split_into_cells(x, left_rank)
        v_cell_right_array = split_into_cells(y, right_rank)

        v_cells = [
            _apply_dyad(v, lx, ry)
            for lx, ry in zip(
                v_cell_left_array.cells, v_cell_right_array.cells, strict=True
            )
        ]
        u_cells = [_apply_monad(u, cell) for cell in v_cells]

        padded_cells = maybe_pad_with_fill_value(u_cells)
        return fill_and_assemble(padded_cells, v_cell_right_array.frame_shape)

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


def ampm_conjunction(left: Verb | Noun, right: Verb | Noun) -> Verb:
    """& conjunction: make a monad from a dyad by providing the left or right noun argument,
    or compose two verbs."""
    if isinstance(left, Noun) and isinstance(right, Verb):
        function = functools.partial(right.dyad.function, left.implementation)
        verb_spelling = maybe_parenthesise_verb_spelling(right.spelling)
        spelling = f"{left.implementation}&{verb_spelling}"
        monad = Monad(name=spelling, rank=INFINITY, function=function)
        dyad = None

    elif isinstance(left, Verb) and isinstance(right, Noun):
        # functools.partial cannot be used to apply to right argument of ufuncs
        # as they do not accept kwargs, so we need to wrap the function.
        def _wrapper(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return _apply_dyad(left, x, y)

        function = functools.partial(_wrapper, y=right.implementation)
        verb_spelling = maybe_parenthesise_verb_spelling(left.spelling)
        spelling = f"{verb_spelling}&{right.implementation}"
        monad = Monad(name=spelling, rank=INFINITY, function=function)
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


def ampdot_conjunction(u: Verb, v: Verb) -> Verb:
    """&. conjunction: u&.v is equivalent to (u&.:v)"mv , where mv is the monadic rank of v."""
    if v.monad is None:
        raise ValenceError(f"{v.spelling} has no monadic form")
    verb = ampdotco_conjunction(u, v)
    return _modify_rank(verb, v.monad.rank)


def hatco_conjunction(u: Verb, noun_or_verb: Noun | Verb) -> Verb:
    """^: conjunction: power of verb."""

    if isinstance(noun_or_verb, Verb):
        raise JinxNotImplementedError("^: conjunction with verb is not yet implemented")

    if isinstance(noun_or_verb, Noun):
        exponent: Noun = noun_or_verb

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
            monad=Monad(
                name="SQUARELF",
                rank=INFINITY,
                function=lambda y: y,
            ),
            dyad=Dyad(
                name="SQUARELF",
                left_rank=INFINITY,
                right_rank=INFINITY,
                function=lambda x, y: y,
            ),
            obverse="]",
        )

    # Special case (^:1) is u
    if (
        np.isscalar(exponent.implementation) or exponent.implementation.shape == ()
    ) and exponent == 1:
        return u

    if np.isinf(exponent.implementation).any():
        raise JinxNotImplementedError(
            "^: with infinite exponent is not yet implemented"
        )

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
        array = np.asarray(result)
        return array.reshape(exponent.implementation.shape + result[0].shape)

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
        array = np.asarray(result)
        return array.reshape(exponent.implementation.shape + result[0].shape)

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


CONJUNCTION_MAP = {
    "RANK": rank_conjunction,
    "AT": at_conjunction,
    "ATCO": atco_conjunction,
    "AMPM": ampm_conjunction,
    "AMPDOT": ampdot_conjunction,
    "AMPDOTCO": ampdotco_conjunction,
    "HATCO": hatco_conjunction,
}
