"""Methods implementing J adverbs."""

import functools
from typing import Callable

import jax
import jax.numpy as jnp
from jinx.errors import JinxNotImplementedError, ValenceError
from jinx.execution.jax.application import _apply_dyad
from jinx.primitives import PRIMITIVE_ADVERB_MAP
from jinx.vocabulary import Dyad, EntityExecutedAdverb, Monad, Verb

INFINITY = float("inf")


def slash_adverb(verb: Verb[jax.Array]) -> Verb[jax.Array]:
    if verb.dyad is None or verb.dyad.function is None:
        # Note: this differs from J which still allows the adverb to be applied
        # to a verb, but may raise an error when the new verb is applied to a noun
        # and the verb has no dyadic valence.
        raise ValenceError(f"Verb {verb.spelling} has no dyadic valence.")

    if isinstance(verb.dyad.function, jnp.ufunc) and verb.dyad.is_commutative:
        monad = verb.dyad.function.reduce
        dyad = verb.dyad.function.outer

    else:
        # Slow path: dyad is not a ufunc.
        # The function is either callable, in which cases it is applied directly,
        # or a Verb object that needs to be applied indirectly with _apply_dyad().
        if isinstance(verb.dyad.function, Verb):
            func = functools.partial(_apply_dyad, verb)  # type: ignore[assignment]
        else:
            func = verb.dyad.function  # type: ignore[assignment]

        def _dyad_arg_swap(x: jax.Array, y: jax.Array) -> jax.Array:
            return func(y, x)

        def _reduce(y: jax.Array) -> jax.Array:
            y = jnp.atleast_1d(y)
            y = jnp.flip(y, axis=0)
            return functools.reduce(_dyad_arg_swap, y)

        monad = _reduce  # type: ignore[assignment]
        dyad = NotImplemented

    return Verb[jax.Array](
        monad=Monad(rank=INFINITY, function=monad),
        dyad=Dyad(left_rank=INFINITY, right_rank=INFINITY, function=dyad),
        entity_type=EntityExecutedAdverb(verb, PRIMITIVE_ADVERB_MAP["SLASH"]),
    )


def bslash_adverb(verb: Verb[jax.Array]) -> Verb[jax.Array]:
    # Common cases that have a straightforward optimisation.
    SPECIAL_MONAD = {
        "+/": jnp.cumulative_sum,
        "*/": jnp.cumulative_prod,
    }

    if verb.spelling in SPECIAL_MONAD:
        monad_ = SPECIAL_MONAD[verb.spelling]

    else:
        raise JinxNotImplementedError(
            f"Adverb \\ applied to verb {verb.spelling} is not yet implemented."
        )

    return Verb(
        monad=Monad(rank=INFINITY, function=monad_),
        dyad=Dyad(left_rank=0, right_rank=INFINITY, function=None),  # type: ignore[arg-type]
        entity_type=EntityExecutedAdverb(verb, PRIMITIVE_ADVERB_MAP["BSLASH"]),
    )


ADVERB_MAP: dict[str, Callable[[Verb[jax.Array]], Verb[jax.Array]]] = {
    "SLASH": slash_adverb,
    "BSLASH": bslash_adverb,
}
