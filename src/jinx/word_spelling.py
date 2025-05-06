"""Word Spelling.

Given constituent words of a sentence, associate it a part of speech (noun,
verb, adverb, etc.).

Parsing of numeric constants is also done here.

If a word does not map to a recognised part of speech, raise a SpellingError.

"""

import re

from jinx.errors import SpellingError
from jinx.primitives import PRIMITIVES
from jinx.vocabulary import (
    Word,
    Atom,
    PartOfSpeechT,
    Array,
    DataType,
    Comment,
    Name,
    Punctuation,
)


def parse_integer(word: str) -> int | None:
    # Integers are a sequence of digits, optionally terminated
    # by a single decimal point (which can be followed by any number
    # of trailing 0s), or a single 'x' (denoting extended precision)
    # and ignored here as all Python integers are extended precision).
    match = re.fullmatch(r"_?(\d+)(?:x|\.0*)?", word)
    if not match:
        return None

    value = int(match.group(1))
    return -value if word.startswith("_") else value


def parse_float(word: str) -> float | None:
    if word == "_":
        return float("inf")
    if word == "__":
        return float("-inf")
    if not re.fullmatch(r"_?\d+\.\d*", word):
        return None
    if word.startswith("_"):
        return -float(word[1:])
    return float(word)


def spell_numeric(word: Word) -> Atom | Array:
    values = word.value.split()
    numbers = []
    data_type = DataType.Integer

    for value in values:
        int_value = parse_integer(value)
        if int_value is not None:
            numbers.append(int_value)
            continue

        float_value = parse_float(value)
        if float_value is not None:
            numbers.append(float_value)
            data_type = DataType.Float
            continue

        raise SpellingError(f"Ill-formed number: {value}")

    if len(numbers) == 1:
        return Atom(data_type=data_type, data=numbers[0])

    return Array(data_type=data_type, data=numbers)


def spell_quoted(word: Word) -> Atom | Array:
    assert word.value[0] == "'" and word.value[-1] == "'"

    data = word.value[1:-1]
    if len(data) <= 1:
        return Atom(data_type=DataType.Byte, data=data)
    return Array(data_type=DataType.Byte, data=data)


PRIMITIVE_MAP = {primitive.spelling: primitive for primitive in PRIMITIVES}

PUNCTUATION_MAP = {
    "(": Punctuation(spelling="(", name="LPAREN"),
    ")": Punctuation(spelling=")", name="RPAREN"),
    "'": Punctuation(spelling="'", name="QUOTE"),
}


def spell(word: Word) -> PartOfSpeechT:
    if word.value in PRIMITIVE_MAP:
        return PRIMITIVE_MAP[word.value]

    if word.value in PUNCTUATION_MAP:
        return PUNCTUATION_MAP[word.value]

    if word.is_numeric:
        return spell_numeric(word)

    if word.value[0] == "'" and word.value[-1] == "'":
        return spell_quoted(word)

    if word.value.startswith("NB."):
        return Comment(spelling=word.value)

    if word[0].isalpha():
        # Only a restricted form of simple names are supported (alphanumeric).
        # This is to avoid the complexity of parsing J's full name grammar which
        # has rules for underscores to support locatives (locales / namespaces).
        #
        # See: https://code.jsoftware.com/wiki/Vocabulary/Locales#Locatives
        if not word.value.isalnum():
            raise SpellingError(f"Only alphanumeric names are supported: {word.value}")
        return Name(spelling=word.value)

    raise SpellingError(f"Unrecognised word: {word}")


def spell_words(words: list[Word]) -> list[PartOfSpeechT]:
    return [spell(word) for word in words]
