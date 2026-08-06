"""Trains: hooks and forks"""

import jax
from jinx.execution.jax.application import _apply_dyad, _apply_monad
from jinx.vocabulary import Dyad, Monad, Noun, Verb

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

    if isinstance(f, Verb):
        f_spelling = f"({f.spelling})" if " " in f.spelling else f.spelling
    else:
        f_spelling = str(f.implementation)

    g_spelling = f"({g.spelling})" if " " in g.spelling else g.spelling
    h_spelling = f"({h.spelling})" if " " in h.spelling else h.spelling
    spelling = f"{f_spelling} {g_spelling} {h_spelling}"

    return Verb[jax.Array](
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
