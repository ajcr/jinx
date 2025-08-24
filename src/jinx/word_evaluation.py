"""Parsing and evaluation.

In J, parsing and evaluation happen simultaneously.

https://www.jsoftware.com/ioj/iojSent.htm#Parsing
https://www.jsoftware.com/help/jforc/parsing_and_execution_ii.htm
https://code.jsoftware.com/wiki/Vocabulary/Modifiers

"""

from jinx.vocabulary import (
    PartOfSpeechT,
    Verb,
    Adverb,
    Conjunction,
    Punctuation,
    Noun,
    Name,
    Comment,
    Copula,
    Atom,
    Array,
)

from jinx.execution.application import (
    apply_monad,
    apply_dyad,
    apply_conjunction,
    apply_adverb,
    build_fork,
    build_hook,
)
from jinx.errors import JValueError
from jinx.execution.conversion import ensure_noun_implementation
from jinx.primitives import PRIMITIVES
from jinx.execution.printing import noun_to_string
from jinx.execution.primitives import PRIMITIVE_MAP
from jinx.word_formation import form_words
from jinx.word_spelling import spell_words


class EvaluationError(Exception):
    pass


def str_(word: Atom | Array | Verb | Conjunction | Adverb) -> str:
    if isinstance(word, str):
        return word
    if isinstance(word, Atom | Array):
        return noun_to_string(word)
    elif isinstance(word, Verb | Adverb | Conjunction):
        return word.spelling
    elif isinstance(word, Name):
        return ""
    else:
        raise NotImplementedError(f"Cannot print word of type {type(word)}")


def print_words(words: list[PartOfSpeechT]) -> None:
    print(" ".join(str_(word) for word in words if word is not None))


def evaluate_single_verb_sentence(sentence: str) -> Verb:
    tokens = form_words(sentence)
    words = spell_words(tokens)
    words = _evaluate_words(words)
    assert len(words) == 2 and isinstance(words[1], Verb)
    return words[1]


def build_verb_noun_phrase(
    words: list[Verb | Noun | Adverb | Conjunction],
) -> Verb | Noun | None:
    """Build the verb or noun phrase from a list of words, or raise an error."""
    while len(words) > 1:
        match words:
            case [left, Adverb(), *remaining]:
                result = apply_adverb(left, words[1])
                words = [result, *remaining]

            case [left, Conjunction(), right, *remaining]:
                result = apply_conjunction(left, words[1], right)
                words = [result, *remaining]

            case _:
                raise EvaluationError("Unable to build verb/noun phrase")

    if not words:
        return None

    if isinstance(words[0], Verb | Noun):
        return words[0]

    raise EvaluationError("Unable to build verb/noun phrase")


def evaluate_words(words: list[PartOfSpeechT], level: int = 0, variables: dict[Name, PartOfSpeechT] | None = None) -> list[PartOfSpeechT]:
    # Ensure noun and verb implementations are set according to the chosen execution
    # framework (this is just NumPy for now).
    for word in words:
        if isinstance(word, Noun):
            ensure_noun_implementation(word)

    for primitive in PRIMITIVES:
        if primitive.name not in PRIMITIVE_MAP:
            continue
        if isinstance(primitive, Verb):
            monad, dyad = PRIMITIVE_MAP[primitive.name]
            if primitive.monad is not None:
                primitive.monad.function = monad
            if primitive.dyad is not None:
                primitive.dyad.function = dyad
        if isinstance(primitive, (Adverb, Conjunction)):
            primitive.function = PRIMITIVE_MAP[primitive.name]

    # Verb obverses are converted from strings to Verb objects.
    for word in words:
        if isinstance(word, Verb) and isinstance(word.obverse, str):
            verb = evaluate_single_verb_sentence(word.obverse)
            word.obverse = verb

    return _evaluate_words(words, level=level, variables=variables)


def get_parts_to_left(
    word: PartOfSpeechT, words: list[PartOfSpeechT | str], current_level: int
) -> list[PartOfSpeechT]:
    """Get the parts of speach to the left of the current word, modifying list of remaining words.

    This method is called when the last word we encountered is an adverb or conjunction and
    a verb or noun phrase is expected to the left of it.
    """
    parts_to_left = []

    while words:
        # A verb/noun phrase starts with a verb/noun which does not have a conjunction to its left.
        if isinstance(word, Noun | Verb):
            if not isinstance(words[-1], Conjunction):
                parts_to_left = [word, *parts_to_left]
                break
            else:
                conjunction = words.pop()
                parts_to_left = [conjunction, word, *parts_to_left]

        elif isinstance(word, Adverb | Conjunction):
            parts_to_left = [word, *parts_to_left]

        elif word == ")":
            word = evaluate_words(words, level=current_level + 1)
            continue

        else:
            break

        if words:
            word = words.pop()
            if isinstance(word, (Punctuation, Copula)):
                word = word.spelling

    return parts_to_left


def _fragment_name_to_parts_of_speech(
    fragment: list[PartOfSpeechT | str], variables: dict[Name, PartOfSpeechT]
) -> list[PartOfSpeechT]:
    new_fragment = []
    for word in fragment:
        if isinstance(word, Name):
            if word.spelling in variables:
                new_fragment.append(variables[word.spelling])
            else:
                # raise JValueError(f"Value error: {word.spelling}")
                new_fragment.append(word.spelling)
        else:
            new_fragment.append(word)
    return new_fragment


def _evaluate_words(
    words: list[PartOfSpeechT],
    level: int = 0,
    variables: dict[Name, PartOfSpeechT] | None = None,
) -> list[PartOfSpeechT]:
    if variables is None:
        variables = {}

    # If the first word is None, prepend a None to the list denoting the left-most
    # edge of the expression.
    if words[0] is not None:
        words = [None, *words]

    fragment = []

    while words:
        word = words.pop()

        if isinstance(word, Comment):
            continue

        elif isinstance(word, (Punctuation, Copula)):
            word = word.spelling

        # If the next word closes a parenthesis, we need to evaluate the words inside it
        # first to get the next word to prepend to the fragment.
        if word == ")":
            word = evaluate_words(words, level=level + 1, variables=variables)

        # If the fragment has a modifier (adverb/conjunction) at the start, we need to find the
        # entire verb/noun phrase to the left as the next word to prepend to the fragment.
        # Contrary to usual parsing and evaluation, the verb/noun phrase is evaluated left-to-right.
        if fragment and isinstance(fragment[0], Adverb | Conjunction):
            parts_to_left = get_parts_to_left(word, words, level)
            word = build_verb_noun_phrase(parts_to_left)

        fragment = [word, *fragment]

        # fmt: off
        while True:

            # Replace any variables in the fragment with their values.
            # TODO: name the variable something else - want to look up the variable value each time.
            fragment = _fragment_name_to_parts_of_speech(fragment, variables)

            match fragment:

                # 0. Monad
                case None | "=." | "=:" | "(", Verb(), Noun():
                    edge, verb, noun = fragment
                    result = apply_monad(verb, noun)
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 1. Monad
                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Adverb() | Verb(), Verb(), Noun():
                    edge, _, verb, noun = fragment
                    result = apply_monad(verb, noun)
                    fragment[2:] = [result]

                # 2. Dyad
                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Noun(), Verb(), Noun():
                    edge, noun, verb, noun_2 = fragment
                    result = apply_dyad(verb, noun, noun_2)
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 3. Adverb
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb() | Noun(),
                    Adverb(),
                    *_,
                ):
                    edge, verb, adverb, *last = fragment
                    result = apply_adverb(verb, adverb)
                    if edge == "(" and last == [")"] and level > 0:
                        return result
                    fragment[1:3] = [result]

                # 4. Conjunction
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb() | Noun(),
                    Conjunction(),
                    Verb() | Noun(),
                    *_,
                ):
                    edge, verb_or_noun_1, conjunction, verb_or_noun_2, *last = fragment
                    result = apply_conjunction(verb_or_noun_1, conjunction, verb_or_noun_2)
                    if edge == "(" and last == [")"] and level > 0:
                        return result
                    fragment[1:4] = [result]

                # 5. Fork
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb() | Noun(),
                    Verb(),
                    Verb(),
                ):
                    edge, verb_or_noun_1, verb_2, verb_3 = fragment
                    result = build_fork(verb_or_noun_1, verb_2, verb_3)
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:4] = [result]

                # 6. Hook/Adverb
                case (
                    None | "=." | "=:" | "(",
                    Conjunction() | Adverb() | Verb() | Noun(),
                    Conjunction() | Adverb() | Verb() | Noun(),
                    *_,
                ):
                    edge, cavn1, cavn2, *last = fragment
                    if isinstance(cavn1, Verb) and isinstance(cavn2, Verb):
                        result = build_hook(cavn1, cavn2)
                    else:
                        raise NotImplementedError("Only VV is implemented for hook/adverb matching")
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 7. Is
                case Name(), "=." | "=:", Conjunction() | Adverb() | Verb() | Noun(), *_:
                    name, _, cavn, *last = fragment
                    variables[name] = cavn
                    fragment = [name, *last]

                # 8. Parentheses
                # Differs from the J source as it does not match ")" and instead checks
                # the level to ensure that "(" is balanced.
                case ["(", Conjunction() | Adverb() | Verb() | Noun()]:
                    _, cavn = fragment
                    if level > 0:
                        return cavn
                    raise EvaluationError("Unbalanced parentheses")

                # Non-executable fragment.
                case _:
                    break

        # fmt: on

    if len(fragment) > 2:
        raise EvaluationError("Unexecutable fragment")

    return fragment
