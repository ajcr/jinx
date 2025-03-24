"""Primitives.

Names of sequences of characters that are assigned some meaning in the J language.

These objects are not tied to their implementation here and are just used for
spelling (recognition of fragments of a sentence).

See: https://code.jsoftware.com/wiki/NuVoc
"""

from vocabulary import Verb, Adverb, Conjunction, Copula

PRIMITIVES: list[Verb | Adverb | Conjunction | Copula] = [
    Verb("=", "EQ"),
    Copula("=.", "EQDOT"),
    Copula("=:", "EQCO"),
    Verb("<", "LT"),
    Verb("<.", "LTDOT"),
    Verb("<:", "LTCO"),
    Verb(">", "GT"),
    Verb(">.", "GTDOT"),
    Verb(">:", "GTCO"),
    Verb("+", "PLUS"),
    Verb("+.", "PLUSDOT"),
    Verb("+:", "PLUSCO"),
    Verb("*", "STAR"),
    Verb("*.", "STARDOT"),
    Verb("*:", "STARCO"),
    Verb("-", "MINUS"),
    Verb("-.", "MINUSDOT"),
    Verb("-:", "MINUSCO"),
    Verb("%", "PRECENT"),
    Verb("%.", "PRECENTDOT"),
    Verb("%:", "PERCENTCO"),
    Adverb("/", "SLASH"),
    Adverb("/.", "SLASHDOT"),
    Verb("/:", "SLASHCO"),
    Conjunction("@", "AT"),
    # More to be implemented...
]
