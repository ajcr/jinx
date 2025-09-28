from dataclasses import dataclass
from typing import Callable

from jinx.vocabulary import Adverb, Conjunction, Noun, Verb


@dataclass(frozen=True)
class Executor[T]:
    apply_monad: Callable[[Verb[T], Noun[T]], Noun[T]]
    """Apply monadic form of verb to a noun."""

    apply_dyad: Callable[[Verb[T], Noun[T], Noun[T]], Noun[T]]
    """Apply dyadic form of verb to two nouns."""

    apply_conjunction: Callable[
        [Verb[T] | Noun[T], Conjunction, Verb[T]], Verb[T] | Noun[T]
    ]
    """Apply conjunction to left and right arguments."""

    apply_adverb: Callable[[Verb[T] | Noun[T], Adverb], Verb[T] | Noun[T]]
    """Apply adverb to left argument."""

    build_fork: Callable[[Noun[T] | Verb[T], Verb[T], Verb[T]], Verb[T]]
    """Build fork."""

    build_hook: Callable[[Verb[T], Verb[T]], Verb[T]]
    """Build hook."""

    ensure_noun_implementation: Callable[[Noun[T]], None]
    """Ensure that the noun has an implementation."""

    primitive_verb_map: dict[
        str, tuple[Callable[[T], T] | None, Callable[[T, T], T] | None]
    ]
    """Map of primitive verb names to implementations of monad and dyad functions."""

    primitive_adverb_map: dict[str, Callable[[Verb[T]], Verb[T]]]
    """Map of primitive adverb names to implementation function."""

    primitive_conjuction_map: dict[
        str, Callable[[Verb[T] | Noun[T], Verb[T] | Noun[T]], Verb[T]]
    ]
    """Map of primitive conjunction names to implementation function."""

    noun_to_string: Callable[[Noun[T]], str]
    """Convert a noun to a string representation for printing."""


def load_executor(name: str) -> Executor:
    if name == "numpy":
        from jinx.execution.numpy import executor

        return executor

    raise ValueError(f"Unknown executor: {name}")
