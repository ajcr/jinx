"""Trains: hooks and forks"""

import numpy as np
from jinx.execution.numpy.application import _apply_dyad, _apply_monad
from jinx.vocabulary import Dyad, Monad, Noun, Verb

INFINITY = float("inf")


def build_hook(f: Verb[np.ndarray], g: Verb[np.ndarray]) -> Verb[np.ndarray]:
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

    return Verb[np.ndarray](
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


def build_fork(
    f: Verb[np.ndarray] | Noun[np.ndarray], g: Verb[np.ndarray], h: Verb[np.ndarray]
) -> Verb[np.ndarray]:
    """Build a fork given verbs f, g, h.

      (f g h) y  ->    (f y) g   (h y)
    x (f g h) y  ->  (x f y) g (x h y)

    The new verb has infinite rank.

    Note that f can be a noun, in which case there is one fewer function calls.
    """

    def _monad(y: np.ndarray) -> np.ndarray:
        if isinstance(f, Verb) and f.spelling == "[:":
            hy = _apply_monad(h, y)
            return _apply_monad(g, hy)

        if isinstance(f, Verb):
            a = _apply_monad(f, y)
        else:
            a = f.implementation
        b = _apply_monad(h, y)
        return _apply_dyad(g, a, b)

    def _dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

    return Verb[np.ndarray](
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
