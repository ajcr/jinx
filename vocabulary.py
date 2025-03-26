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
from typing import NamedTuple


class Word(NamedTuple):
    "Sequence of characters that can be recognised as a part of the J language."

    value: str
    is_numeric: bool
    start: int
    end: int


class DataType(Enum):
    Integer = auto()
    Float = auto()
    Byte = auto()


@dataclass
class Noun:
    pass


@dataclass
class Atom(Noun):
    data_type: DataType
    data: int | float | str


@dataclass
class Array(Noun):
    data_type: DataType
    data: list[int | float] | str


@dataclass
class Verb:
    spelling: str
    name: str


@dataclass
class Adverb:
    spelling: str
    name: str


@dataclass
class Conjunction:
    spelling: str
    name: str


@dataclass
class Copula:
    spelling: str
    name: str


@dataclass
class Punctuation:
    spelling: str
    name: str


@dataclass
class Comment:
    spelling: str


@dataclass
class Name:
    spelling: str


PunctuationT = Punctuation | Comment
PartOfSpeechT = Noun | Verb | Adverb | Conjunction | PunctuationT | Copula | Name
