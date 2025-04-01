"""Parsing and evaluation.

In J, parsing and evaluation happen simultaneously.

https://www.jsoftware.com/ioj/iojSent.htm#Parsing
https://www.jsoftware.com/help/jforc/parsing_and_execution_ii.htm

"""

from vocabulary import (
    PartOfSpeechT,
    Verb,
    Adverb,
    Conjunction,
    Punctuation,
    Noun,
    Name,
    Comment,
    Copula,
)

from np_implementation import PRIMITIVE_MAP, convert_noun_np, ndarray_or_scalar_to_noun


def ensure_noun_implementation(noun: Noun) -> None:
    if noun.implementation is None:
        noun.implementation = convert_noun_np(noun)


def evaluate(words: list[PartOfSpeechT]) -> list[PartOfSpeechT]:
    """Evaluate the words in the sentence."""

    fragment = []
    words = [None, *words]

    verb: Verb
    noun: Noun
    noun_2: Noun
    adverb: Adverb

    while words:
        word = words.pop()

        if isinstance(word, Comment):
            continue

        elif isinstance(word, (Punctuation, Copula)):
            word = word.spelling

        # TODO: If word is a Name, lookup the Verb/Noun it refers to and add that.

        fragment = [word, *fragment]

        while True:
            match fragment:
                # 0. Monad
                case [None | "=." | "=:" | "(", Verb(), Noun()] | [
                    None | "=." | "=:" | "(",
                    Verb(),
                    Noun(),
                    _,
                ]:
                    _, verb, noun = fragment
                    ensure_noun_implementation(noun)
                    f = PRIMITIVE_MAP[verb.name][0]
                    result = f(noun)(noun.implementation)
                    fragment[1:] = [ndarray_or_scalar_to_noun(result)]

                # 1. Monad
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb(),
                    Verb(),
                    Noun(),
                ):
                    _, _, verb, noun = fragment
                    ensure_noun_implementation(noun)
                    f = PRIMITIVE_MAP[verb.name][0]
                    result = f(noun)(noun.implementation)
                    fragment[2:] = [ndarray_or_scalar_to_noun(result)]

                # 2. Dyad
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Noun(),
                    Verb(),
                    Noun(),
                ):
                    _, noun, verb, noun_2 = fragment
                    ensure_noun_implementation(noun)
                    ensure_noun_implementation(noun_2)
                    f = PRIMITIVE_MAP[verb.name][1]
                    result = f(noun, noun_2)(noun.implementation, noun_2.implementation)
                    fragment[1:] = [ndarray_or_scalar_to_noun(result)]

                # 3. Adverb
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb() | Noun(),
                    Adverb(),
                    _,
                ):
                    raise NotImplementedError("adverb")

                # 4. Conjunction
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb() | Noun(),
                    Conjunction(),
                    Verb() | Noun(),
                ):
                    raise NotImplementedError("conjunction")

                # 5. Fork
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb() | Noun(),
                    Verb(),
                    Verb(),
                ):
                    raise NotImplementedError("fork")

                # 6. Hook/Adverb
                case (
                    None | "=." | "=:" | "(",
                    Conjunction() | Adverb() | Verb() | Noun(),
                    Conjunction() | Adverb() | Verb() | Noun(),
                    _,
                ):
                    raise NotImplementedError("hook/adverb")

                # 7. Is
                case Name(), "=." | "=:", Conjunction() | Adverb() | Verb() | Noun(), _:
                    raise NotImplementedError("copula")

                # 8. Parentheses
                case "(", Conjunction() | Adverb() | Verb() | Noun(), ")", _:
                    raise NotImplementedError("parentheses")

                # Non-executable fragment.
                case _:
                    break

    return fragment
