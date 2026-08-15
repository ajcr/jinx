import pytest

from jinx.primitives import PRIMITIVE_MAP
from jinx.vocabulary import (
    EntityExecutedAdverb,
    EntityExecutedConjunction,
    EntityHook,
    Verb,
)

# +/
VERB_PLUS_SLASH = Verb(
    entity_type=EntityExecutedAdverb(
        v0=PRIMITIVE_MAP["PLUS"], a1=PRIMITIVE_MAP["SLASH"]
    ),
)

# *&(+/)
VERB_STAR_AMPM_VERB_PLUS_SLASH = Verb(
    entity_type=EntityExecutedConjunction(
        v0=PRIMITIVE_MAP["STAR"], c1=PRIMITIVE_MAP["AMPM"], v2=VERB_PLUS_SLASH
    ),
)

# +/@(*&,)
VERB_PLUS_SLASH_AT_VERB_STAR_AMPM_COMMA = Verb(
    entity_type=EntityExecutedConjunction(
        v0=VERB_PLUS_SLASH,
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
VERB_STAR_VERB_PLUS = Verb(
    entity_type=EntityHook(v0=PRIMITIVE_MAP["STAR"], v1=PRIMITIVE_MAP["PLUS"])
)

# (+/ *)
VERB_PLUS_SLASH_VERB_STAR = Verb(
    entity_type=EntityHook(v0=VERB_PLUS_SLASH, v1=PRIMITIVE_MAP["STAR"])
)

# * +@-/
VERB_STAR_VERB_PLUS_AT_PLUS_SLASH = Verb(
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


@pytest.mark.parametrize(
    "verb, expected_spelling",
    [
        pytest.param(VERB_PLUS_SLASH, "+/", id="+/"),
        pytest.param(VERB_STAR_AMPM_VERB_PLUS_SLASH, "*&(+/)", id="*&(+/)"),
        pytest.param(
            VERB_PLUS_SLASH_AT_VERB_STAR_AMPM_COMMA, "+/@(*&,)", id="+/@(*&,)"
        ),
        pytest.param(VERB_STAR_VERB_PLUS, "* +", id="* +"),
        pytest.param(VERB_PLUS_SLASH_VERB_STAR, "+/ *", id="+/ *"),
        pytest.param(VERB_STAR_VERB_PLUS_AT_PLUS_SLASH, "* +@-/", id="* +@-/"),
    ],
)
def test_verb_spelling(verb, expected_spelling):
    assert str(verb) == expected_spelling
