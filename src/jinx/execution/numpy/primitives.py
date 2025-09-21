"""J primitives."""

from jinx.execution.numpy.adverbs import ADVERB_MAP
from jinx.execution.numpy.conjunctions import CONJUNCTION_MAP
from jinx.execution.numpy.verbs import VERB_MAP


PRIMITIVE_MAP = ADVERB_MAP | CONJUNCTION_MAP | VERB_MAP
