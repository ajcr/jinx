"""J primitives."""

from jinx.execution.adverbs import ADVERB_MAP
from jinx.execution.conjunctions import CONJUNCTION_MAP
from jinx.execution.verbs import VERB_MAP


PRIMITIVE_MAP = ADVERB_MAP | CONJUNCTION_MAP | VERB_MAP
