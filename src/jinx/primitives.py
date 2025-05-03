"""Primitives.

Names of sequences of characters that are assigned some meaning in the J language.

These objects are not tied to their implementation here and are just used for
spelling (recognition of fragments of a sentence).

See: https://code.jsoftware.com/wiki/NuVoc
"""

from jinx.vocabulary import Verb, Adverb, Conjunction, Copula, Monad, Dyad

# The monad and dyad fields are not populated here (they default to None).
# This is to allow the correct implementation of the verb to be looked up
# when the verb is evaluated in the context of a sentence. It also allows
# different implementations of a verb to be chosen (pure Python, NumPy, etc.)

INFINITY = float("inf")

PRIMITIVES: list[Verb | Adverb | Conjunction | Copula] = [
    Verb("=", "EQ", dyad=Dyad(name="Equal", left_rank=0, right_rank=0)),
    Copula("=.", "EQDOT"),
    Copula("=:", "EQCO"),
    Verb(
        "<.",
        "LTDOT",
        monad=Monad(name="Floor", rank=0),
        dyad=Dyad(name="Min", left_rank=0, right_rank=0),
    ),
    Verb(
        ">.",
        "GTDOT",
        monad=Monad(name="Ceiling", rank=0),
        dyad=Dyad(name="Max", left_rank=0, right_rank=0),
    ),
    Verb(">:", "GTCO"),
    Verb(
        "+",
        "PLUS",
        monad=Monad(name="Conjugate", rank=0),
        dyad=Dyad(name="Plus", left_rank=0, right_rank=0),
    ),
    Verb(
        "+:",
        "PLUSCO",
        monad=Monad(name="Double", rank=0),
        dyad=Dyad(name="Not-Or", left_rank=0, right_rank=0, is_commutative=True),
    ),
    Verb(
        "*",
        "STAR",
        monad=Monad(name="Signum", rank=0),
        dyad=Dyad(name="Times", left_rank=0, right_rank=0),
    ),
    Verb(
        "*:",
        "STARCO",
        monad=Monad(name="Square", rank=0),
        dyad=Dyad(name="Not-And", left_rank=0, right_rank=0, is_commutative=True),
    ),
    Verb(
        "-",
        "MINUS",
        monad=Monad(name="Negate", rank=0),
        dyad=Dyad(name="Minus", left_rank=0, right_rank=0, is_commutative=False),
    ),
    Verb(
        "%",
        "PERCENT",
        monad=Monad(name="Reciprocal", rank=0),
        dyad=Dyad(name="Divide", left_rank=0, right_rank=0, is_commutative=False),
    ),
    Verb(
        "%:",
        "PERCENTCO",
        monad=Monad(name="Square Root", rank=0),
        dyad=Dyad(name="Root", left_rank=0, right_rank=0),
    ),
    Verb(
        "^",
        "HAT",
        monad=Monad(name="Exponential", rank=0),
        dyad=Dyad(name="Power", left_rank=0, right_rank=0, is_commutative=False),
    ),
    Verb(
        "^.",
        "HATDOT",
        monad=Monad(name="Natural Log", rank=0),
        dyad=Dyad(name="Logarithm", left_rank=0, right_rank=0, is_commutative=False),
    ),
    Verb(
        "<.",
        "LTDOT",
        monad=Monad(name="Floor", rank=0),
        dyad=Dyad(name="Min", left_rank=0, right_rank=0),
    ),
    Verb(
        ">.",
        "GTDOT",
        monad=Monad(name="Ceiling", rank=0),
        dyad=Dyad(name="Max", left_rank=0, right_rank=0),
    ),
    Verb(
        "<:",
        "LTCO",
        monad=Monad(name="Decrement", rank=0),
        dyad=Dyad(name="Less Or Equal", left_rank=0, right_rank=0),
    ),
    Verb(
        ">:",
        "GTCO",
        monad=Monad(name="Increment", rank=0),
        dyad=Dyad(name="Larger Or Equal", left_rank=0, right_rank=0),
    ),
    Verb(
        "~.",
        "TILDEDOT",
        monad=Monad(name="Nub", rank=INFINITY),
    ),
    Verb(
        "$",
        "DOLLAR",
        monad=Monad(name="Shape Of", rank=INFINITY),
        dyad=Dyad(name="Shape", left_rank=1, right_rank=INFINITY),
    ),
    Conjunction("@", "AT"),
    Verb(
        "i.",
        "IDOT",
        monad=Monad(name="Integers", rank=1),
        dyad=Dyad(name="Index Of", left_rank=INFINITY, right_rank=INFINITY),
    ),
    Adverb(
        "/",
        "SLASH",
        monad=Monad(name="Insert", rank=INFINITY),
        dyad=Dyad(name="Table", left_rank=INFINITY, right_rank=INFINITY),
    ),
    Conjunction('"', "RANK"),
    Verb(
        ",",
        "COMMA",
        monad=Monad(name="Ravel", rank=INFINITY),
        dyad=Dyad(name="Append", left_rank=INFINITY, right_rank=INFINITY),
    ),
    Verb(
        "|",
        "BAR",
        monad=Monad(name="Magnitude", rank=0),
        dyad=Dyad(name="Residue", left_rank=0, right_rank=0),
    ),
    Verb(
        "|.",
        "BARDOT",
        monad=Monad(name="Reverse", rank=INFINITY),
        dyad=Dyad(name="Rotate", left_rank=1, right_rank=INFINITY),
    ),
]


PRIMITIVE_MAP = {primitive.name: primitive for primitive in PRIMITIVES}
