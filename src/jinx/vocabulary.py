"""J Vocabulary.

Building blocks / parts of speech for the J language.

The objects here are not tied to any implementation details needed for
execution (e.g. a verb is not tied to the code that will execute it).

The objects are just used to tag the words in the sentence so that they
can be evaluated at run time according to the context they are used in.

Resources:
- https://code.jsoftware.com/wiki/Vocabulary/Nouns
- https://code.jsoftware.com/wiki/Vocabulary/Words
- https://code.jsoftware.com/wiki/Vocabulary/Glossary

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, NamedTuple, Sequence

# Rank can be an integer or infinite (a float). It can't be any other float value
# but the type system does not make this easy to express.
RankT = int | float


class Word(NamedTuple):
    """Sequence of characters that can be recognised as a part of the J language."""

    value: str
    """The string value of the word."""

    is_numeric: bool
    """Whether the word represents a numeric value (e.g. an integer or float)."""

    start: int
    """The start index of the word in the expression."""

    end: int
    """The end index of the word in the expression (exclusive, so `expression[start:end]` is the value)."""


class DataType(Enum):
    Integer = auto()
    Float = auto()
    Byte = auto()
    Box = auto()


@dataclass
class Noun[T]:
    data_type: DataType
    """Data type of value."""

    data: Sequence[int | float | str] = field(default_factory=list)
    """Data to represent the value itself, parsed from the word."""

    implementation: T = None  # type: ignore[assignment]
    """Implementation of the noun, e.g. a NumPy array."""


@dataclass
class Monad[T]:
    rank: RankT
    """Rank of monadic valence of the verb."""

    name: str | None = None
    """Name of the monadic verb."""

    function: Callable[[T], T] | Verb[T] = None  # type: ignore[assignment]
    """Function to execute the monadic verb, or another Verb object. Initially
    set to None and then updated at runtime."""


@dataclass
class Dyad[T]:
    left_rank: RankT
    """Left rank of the dyadic verb."""

    right_rank: RankT
    """Right rank of the dyadic verb."""

    name: str | None = None
    """Name of the dyadic verb."""

    function: Callable[[T, T], T] | Verb[T] = None  # type: ignore[assignment]
    """Function to execute the monadic verb, or another Verb object. Initially
    set to None and then updated at runtime."""

    is_commutative: bool = False
    """Whether the dyadic verb is commutative."""


@dataclass
class Adverb[T]:
    spelling: str
    """The symbolic spelling of the adverb, e.g. `/`."""

    name: str | None
    """The name of the adverb, e.g. `SLASH`."""

    monad: Monad[T] | None = None
    """The monadic form of the adverb, if it exists."""

    dyad: Dyad[T] | None = None
    """The dyadic form of the adverb, if it exists."""

    function: Callable[[Verb[T] | Noun[T]], Verb[T]] = None  # type: ignore[assignment]
    """Function of a single argument to implement the adverb."""


@dataclass
class Conjunction[T]:
    spelling: str
    """The symbolic spelling of the conjunction, e.g. `@:`."""

    name: str
    """The name of the conjunction, e.g. `ATCO`."""

    function: Callable[[Verb[T] | Noun[T], Verb[T] | Noun[T]], Verb[T] | Noun[T]] = None  # type: ignore[assignment]
    """Function of a two arguments to implement the conjunction."""


@dataclass
class Copula:
    spelling: str
    """The symbolic spelling of the copula, e.g. `=.`."""

    name: str
    """The name of the copula, e.g. `EQCO`."""


@dataclass
class Punctuation:
    spelling: str
    """The symbolic spelling of the punctuation symbol, e.g. `(`."""

    name: str
    """The name of the punctuation, e.g. `LPAREN`."""


@dataclass
class Comment:
    spelling: str
    """The string value of the comment."""


@dataclass
class Name:
    spelling: str
    """The string value of the name."""


class EntityType:
    """Base class for entity type."""

    def get_spelling(self):
        raise NotImplementedError


@dataclass
class EntityPrimitive(EntityType):
    """Primitive type (defined as part of J)."""

    pass


@dataclass
class EntityReferenceToNamedEntity(EntityType):
    pass


@dataclass
class EntityExecutedConjunction(EntityType):
    """Executed conjunction."""

    x0: Verb | Noun
    c1: Conjunction
    x2: Verb | Noun

    def get_spelling(self) -> str:
        if is_hook(self.x0) or is_fork(self.x0):
            x0_str = f"({self.x0})"
        else:
            x0_str = str(self.x0)
        if isinstance(self.x2, Noun):
            x2_str = str(self.x2.implementation)  # TODO
        elif is_primitive_verb(self.x2):
            x2_str = str(self.x2)
        else:
            x2_str = f"({self.x2})"
        return f"{x0_str}{self.c1.spelling}{x2_str}"


@dataclass
class EntityExecutedAdverb(EntityType):
    """Executed adverb."""

    v0: Verb
    a1: Adverb

    def get_spelling(self) -> str:
        if is_hook(self.v0) or is_fork(self.v0):
            v0_str = f"({self.v0})"
        else:
            v0_str = str(self.v0)
        return f"{v0_str}{self.a1.spelling}"


@dataclass
class EntityHook(EntityType):
    """Hook."""

    v0: Verb
    v1: Verb

    def get_spelling(self) -> str:
        v0_str = maybe_parenthesise_for_train(self.v0)
        if isinstance(self.v1.entity_type, EntityFork):
            v1_str = f"({self.v1})"
        else:
            v1_str = maybe_parenthesise_for_train(self.v1)
        return f"{v0_str} {v1_str}"


@dataclass
class EntityFork(EntityType):
    """Fork."""

    x0: Verb | Noun
    v1: Verb
    v2: Verb

    def get_spelling(self) -> str:
        x0_str = maybe_parenthesise_for_train(self.x0)
        v1_str = maybe_parenthesise_for_train(self.v1)
        if isinstance(self.v2.entity_type, EntityFork):
            v2_str = str(self.v2)
        else:
            v2_str = maybe_parenthesise_for_train(self.v2)
        return f"{x0_str} {v1_str} {v2_str}"


def maybe_parenthesise_for_train(x: Verb | Noun) -> str:
    if (
        is_primitive_verb(x)
        or isinstance(x, Verb)
        and isinstance(x.entity_type, (EntityExecutedAdverb, EntityExecutedConjunction))
    ):
        return str(x)

    return f"({x})"


def is_primitive_verb(x: Any) -> bool:
    return isinstance(x, Verb) and isinstance(x.entity_type, EntityPrimitive)


def is_hook(x: PartOfSpeechT) -> bool:
    return isinstance(x, Verb) and isinstance(x.entity_type, EntityHook)


def is_fork(x: PartOfSpeechT) -> bool:
    return isinstance(x, Verb) and isinstance(x.entity_type, EntityFork)


@dataclass
class Verb[T]:
    spelling: str | None = None
    """The symbolic spelling of the verb, e.g. `+`."""

    name: str | None = None
    """The name of the verb, e.g. `PLUS`, or its spelling if not a primitive J verb."""

    monad: Monad[T] | None = None
    """The monadic form of the verb, if it exists."""

    dyad: Dyad[T] | None = None
    """The dyadic form of the verb, if it exists."""

    obverse: Verb[T] | str | None = None
    """The obverse of the verb, if it exists. This is typically the inverse of the verb."""

    entity_type: EntityType = field(default_factory=EntityPrimitive)
    """The entity type. How the verb was constructed if not primitive."""

    def __str__(self) -> str:
        if is_primitive_verb(self):
            assert self.spelling is not None
            return self.spelling
        return self.entity_type.get_spelling()

    def __repr__(self) -> str:
        return str(self)


PunctuationT = Punctuation | Comment
PartOfSpeechT = Noun | Verb | Adverb | Conjunction | PunctuationT | Copula | Name
