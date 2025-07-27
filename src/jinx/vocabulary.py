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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, NamedTuple


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
class Noun:
    pass


@dataclass(kw_only=True)
class Atom(Noun):
    data_type: DataType
    """Data type of value."""

    data: int | float | str | None = None
    """Data to represent the atom's value, parsed from the word."""

    implementation: Any = None
    """Implementation of the atom, e.g. a NumPy scalar."""


@dataclass(kw_only=True)
class Array(Noun):
    data_type: DataType
    """Data type of values in the array."""

    data: list[int | float] | str | None = None
    """Data to represent the values in the array, parsed from the word."""

    implementation: Any = None
    """Implementation of the atom, e.g. a NumPy array."""


@dataclass
class Monad:
    name: str
    """Name of the monadic verb."""

    rank: int
    """Rank of monadic valence of the verb."""

    function: Callable[[Any], Any] | "Verb" | None = None
    """Function to execute the monadic verb, or another Verb to apply. Initially
    set to and set at runtime."""


@dataclass
class Dyad:
    name: str
    """Name of the dyadic verb."""

    left_rank: int
    """Left rank of the dyadic verb."""

    right_rank: int
    """Right rank of the dyadic verb."""

    function: Callable[[Any, Any], Any] | "Verb" | None = None
    """Function to execute the monadic verb, or another Verb object. Initially
    set to and set at runtime."""

    is_commutative: bool = False
    """Whether the dyadic verb is commutative."""


@dataclass
class Verb:
    spelling: str
    """The symbolic spelling of the verb, e.g. `+`."""

    name: str
    """The name of the verb, e.g. `PLUS`, or its spelling if not a primitive J verb."""

    monad: Monad | None = None
    """The monadic form of the verb, if it exists."""

    dyad: Dyad | None = None
    """The dyadic form of the verb, if it exists."""

    obverse: str | None = None
    """The obverse of the verb, if it exists. This is typically the inverse of the verb."""


@dataclass
class Adverb:
    spelling: str
    """The symbolic spelling of the adverb, e.g. `/`."""

    name: str
    """The name of the adverb, e.g. `SLASH`."""

    monad: Monad | None = None
    """The monadic form of the adverb, if it exists."""

    dyad: Dyad | None = None
    """The dyadic form of the adverb, if it exists."""

    function: Callable[[Any], Any] | None = None
    """Function of a single argument to implement the adverb."""


@dataclass
class Conjunction:
    spelling: str
    """The symbolic spelling of the conjunction, e.g. `@:`."""

    name: str
    """The name of the conjunction, e.g. `ATCO`."""

    function: Callable[[Any, Any], Any] | None = None
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


PunctuationT = Punctuation | Comment
PartOfSpeechT = Noun | Verb | Adverb | Conjunction | PunctuationT | Copula | Name
