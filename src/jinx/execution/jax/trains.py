"""Trains: hooks and forks"""

import jax
from jinx.execution.jax.application import _apply_dyad, _apply_monad
from jinx.vocabulary import Dyad, EntityFork, Monad, Noun, Verb

INFINITY = float("inf")


def build_fork(
    f: Verb[jax.Array] | Noun[jax.Array], g: Verb[jax.Array], h: Verb[jax.Array]
) -> Verb[jax.Array]:
    """Build a fork given verbs f, g, h.

      (f g h) y  ->    (f y) g   (h y)
    x (f g h) y  ->  (x f y) g (x h y)

    The new verb has infinite rank.

    Note that f can be a noun, in which case there is one fewer function calls.
    """

    def _monad(y: jax.Array) -> jax.Array:
        if isinstance(f, Verb) and f.spelling == "[:":
            hy = _apply_monad(h, y)
            return _apply_monad(g, hy)

        if isinstance(f, Verb):
            a = _apply_monad(f, y)
        else:
            a = f.implementation
        b = _apply_monad(h, y)
        return _apply_dyad(g, a, b)

    def _dyad(x: jax.Array, y: jax.Array) -> jax.Array:
        if isinstance(f, Verb) and f.spelling == "[:":
            hy = _apply_dyad(h, x, y)
            return _apply_monad(g, hy)

        if isinstance(f, Verb):
            a = _apply_dyad(f, x, y)
        else:
            a = f.implementation
        b = _apply_dyad(h, x, y)
        return _apply_dyad(g, a, b)

    return Verb[jax.Array](
        monad=Monad(rank=INFINITY, function=_monad),
        dyad=Dyad(left_rank=INFINITY, right_rank=INFINITY, function=_dyad),
        entity_type=EntityFork(f, g, h),
    )
