import pytest

from jinx.primitives import PRIMITIVE_MAP
from jinx.vocabulary import (
    EntityExecutedAdverb,
    EntityExecutedConjunction,
    EntityHook,
    EntityFork,
    Verb,
)

# +/
V_PLUS_SLASH = Verb(
    entity_type=EntityExecutedAdverb(
        v0=PRIMITIVE_MAP["PLUS"], a1=PRIMITIVE_MAP["SLASH"]
    ),
)

# *&(+/)
V_STAR_AMPM_V_PLUS_SLASH = Verb(
    entity_type=EntityExecutedConjunction(
        v0=PRIMITIVE_MAP["STAR"], c1=PRIMITIVE_MAP["AMPM"], v2=V_PLUS_SLASH
    ),
)

# +/@(*&,)
V_PLUS_SLASH_AT_V_STAR_AMPM_COMMA = Verb(
    entity_type=EntityExecutedConjunction(
        v0=V_PLUS_SLASH,
        c1=PRIMITIVE_MAP["AT"],
        v2=Verb(
            entity_type=EntityExecutedConjunction(
                v0=PRIMITIVE_MAP["STAR"],
                c1=PRIMITIVE_MAP["AMPM"],
                v2=PRIMITIVE_MAP["COMMA"],
            ),
        ),
    ),
)

# (* +)
V_STAR_V_PLUS = Verb(
    entity_type=EntityHook(v0=PRIMITIVE_MAP["STAR"], v1=PRIMITIVE_MAP["PLUS"])
)

# (+/ *)
V_PLUS_SLASH_V_STAR = Verb(
    entity_type=EntityHook(v0=V_PLUS_SLASH, v1=PRIMITIVE_MAP["STAR"])
)

# * +@-/
V_STAR_V_PLUS_AT_PLUS_SLASH = Verb(
    entity_type=EntityHook(
        v0=PRIMITIVE_MAP["STAR"],
        v1=Verb(
            entity_type=EntityExecutedAdverb(
                v0=Verb(
                    entity_type=EntityExecutedConjunction(
                        v0=PRIMITIVE_MAP["PLUS"],
                        c1=PRIMITIVE_MAP["AT"],
                        v2=PRIMITIVE_MAP["MINUS"],
                    ),
                ),
                a1=PRIMITIVE_MAP["SLASH"],
            ),
        ),
    ),
)

# Fork (V V V)
V_PLUS_V_PLUS_V_PLUS = Verb(
    entity_type=EntityFork(
        v0=PRIMITIVE_MAP["PLUS"], v1=PRIMITIVE_MAP["PLUS"], v2=PRIMITIVE_MAP["PLUS"]
    ),
)

# Train of 4 verbs, hook (V F)
V_PLUS_V_PLUS_V_PLUS_V_PLUS = Verb(
    entity_type=EntityHook(v0=PRIMITIVE_MAP["PLUS"], v1=V_PLUS_V_PLUS_V_PLUS)
)

# Hook (F V)
V_PLUS_FORK_V_PLUS_V_PLUS_V_PLUS = Verb(
    entity_type=EntityHook(v0=V_PLUS_V_PLUS_V_PLUS, v1=PRIMITIVE_MAP["PLUS"])
)

# Fork (V V F)
VVF = Verb(
    entity_type=EntityFork(
        v0=PRIMITIVE_MAP["PLUS"],
        v1=PRIMITIVE_MAP["PLUS"],
        v2=V_PLUS_V_PLUS_V_PLUS,
    )
)


@pytest.mark.parametrize(
    "verb, expected_spelling",
    [
        pytest.param(V_PLUS_SLASH, "+/", id="+/"),
        pytest.param(V_STAR_AMPM_V_PLUS_SLASH, "*&(+/)", id="*&(+/)"),
        pytest.param(V_PLUS_SLASH_AT_V_STAR_AMPM_COMMA, "+/@(*&,)", id="+/@(*&,)"),
        pytest.param(V_STAR_V_PLUS, "* +", id="* +"),
        pytest.param(V_PLUS_SLASH_V_STAR, "+/ *", id="+/ *"),
        pytest.param(V_STAR_V_PLUS_AT_PLUS_SLASH, "* +@-/", id="* +@-/"),
        pytest.param(V_STAR_V_PLUS_AT_PLUS_SLASH, "* +@-/", id="* +@-/"),
        pytest.param(V_PLUS_V_PLUS_V_PLUS, "+ + +", id="+ + +"),
        pytest.param(V_PLUS_V_PLUS_V_PLUS_V_PLUS, "+ (+ + +)", id="+ + + +"),
        pytest.param(V_PLUS_FORK_V_PLUS_V_PLUS_V_PLUS, "(+ + +) +", id="(+ + +) +"),
        pytest.param(VVF, "+ + + + +", id="+ + + + +"),
    ],
)
def test_verb_spelling(verb, expected_spelling):
    assert str(verb) == expected_spelling
