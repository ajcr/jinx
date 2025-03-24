
"""J Vocabulary

The building blocks of the J language.

"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Literal, NamedTuple


class Word(NamedTuple):
    "Sequence of characters that can be recognised as a part of the J language."
    value: str
    is_numeric: bool
    start: int
    end: int


# https://code.jsoftware.com/wiki/Vocabulary/Nouns
# https://code.jsoftware.com/wiki/Vocabulary/Words#Parts_Of_Speech


class DataType(Enum):
    Integer = auto()
    Float = auto()
    Byte = auto()


@dataclass
class Atom:
    data_type: DataType
    data: int | float | str


@dataclass
class Array:
    data_type: DataType
    data: list[int | float] | str


NounT = Atom | Array
RankT = Literal["-INF"] | int | Literal["+INF"]


@dataclass
class MonadicVerb:
    rank: RankT
    func: Callable[[Atom | Array, Atom | Array], Atom | Array]


@dataclass
class DyadicVerb:
    left_rank: RankT
    right_rank: RankT
    func: Callable[[Atom | Array, Atom | Array], Atom | Array]


@dataclass
class Adverb:
    rank: RankT
    func: Callable[[MonadicVerb], MonadicVerb] | Callable[[DyadicVerb], DyadicVerb]


@dataclass
class Conjunction:
    left_rank: RankT
    right_rank: RankT
    func: Callable[[Atom | Array, Atom | Array], Atom | Array]


@dataclass
class Punctuation:
    value: str


@dataclass
class Comment:
    value: str


NounT = Atom | Array
VerbT = MonadicVerb | DyadicVerb
PunctuationT = Punctuation | Comment

PartOfSpeechT = NounT | VerbT | Adverb | Conjunction | PunctuationT