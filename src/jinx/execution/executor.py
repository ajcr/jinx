from dataclasses import dataclass
from typing import Callable, TypeVar

from jinx.vocabulary import Adverb, Conjunction, Noun, Verb

T = TypeVar("T")


@dataclass(frozen=True)
class Executor:
    apply_monad: Callable[[Verb, Noun], Noun]
    """Apply monadic form of verb to a noun."""

    apply_dyad: Callable[[Verb, Noun, Noun], Noun]
    """Apply dyadic form of verb to two nouns."""

    apply_conjunction: Callable[[Verb | Noun, Conjunction, Verb], Verb | Noun]
    """Apply conjunction to left and right arguments."""

    apply_adverb: Callable[[Verb | Noun, Adverb], Verb | Noun]
    """Apply adverb to left argument."""

    build_fork: Callable[[Noun | Verb, Verb, Verb], Verb]
    """Build fork."""

    build_hook: Callable[[Verb, Verb], Verb]
    """Build hook."""

    ensure_noun_implementation: Callable[[Noun], None]
    """Ensure that the noun has an implementation."""

    primitive_verb_map: dict[
        str, tuple[Callable[[T], T] | None, Callable[[T, T], T] | None]
    ]
    """Map of primitive verb names to implementations of monad and dyad functions."""

    primitive_adverb_map: dict[str, Callable[[Verb], Verb]]
    """Map of primitive adverb names to implementation function."""

    primitive_conjuction_map: dict[str, Callable[[Verb | Noun, Verb | Noun], Verb]]
    """Map of primitive conjunction names to implementation function."""

    noun_to_string: Callable[[Noun], str]
    """Convert a noun to a string representation for printing."""


def load_executor(name: str) -> Executor:
    if name == "numpy":
        from jinx.execution.numpy import executor

        return executor

    raise ValueError(f"Unknown executor: {name}")
