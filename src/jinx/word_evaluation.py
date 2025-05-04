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
    apply_adverb_to_verb,
    build_fork,
    build_hook,
    ensure_verb_implementation,
)
from jinx.execution.conversion import ensure_noun_implementation

from jinx.execution.printing import (
    atom_to_string,
    array_to_string,
)


class EvaluationError(Exception):
    pass


def str_(word: Atom | Array | Verb | Conjunction | Adverb) -> str:
    if isinstance(word, str):
        return word
    if isinstance(word, Atom):
        return atom_to_string(word)
    elif isinstance(word, Array):
        return array_to_string(word)
    elif isinstance(word, (Verb, Conjunction)):
        return word.spelling
    elif isinstance(word, Adverb):
        return word.spelling
    else:
        raise NotImplementedError(f"Cannot print word of type {type(word)}")


def print_words(words: list[PartOfSpeechT]) -> None:
    print(" ".join(str_(word) for word in words if word is not None))


def evaluate_words(words: list[PartOfSpeechT], level: int = 0) -> list[PartOfSpeechT]:
    for word in words:
        if isinstance(word, Noun):
            ensure_noun_implementation(word)
        elif isinstance(word, Verb):
            ensure_verb_implementation(word)

    if words[0] is not None:
        words = [None, *words]

    fragment = []
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

        if word == ")":
            result = evaluate_words(words, level=level + 1)
            fragment = [result, *fragment]

        else:
            fragment = [word, *fragment]

        # fmt: off
        while True:
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

                # Another monad case for a conjunction that: this will be handled by correctly
                # building modifiers in future, see: https://code.jsoftware.com/wiki/Vocabulary/Modifiers
                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Conjunction(), Noun(), Verb(), Noun():
                    edge, conjunction, noun, verb, noun_2 = fragment
                    result = apply_monad(verb, noun_2)
                    fragment[3:] = [result]

                # 2. Dyad
                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Noun(), Verb(), Noun():
                    edge, noun, verb, noun_2 = fragment
                    result = apply_dyad(verb, noun, noun_2)
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 3. Adverb
                case (
                    [None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Verb(), Adverb()] | 
                    [None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Verb(), Adverb(), *_]
                ):
                    edge, verb, adverb, *last = fragment

                    if isinstance(edge, Adverb):
                        # TODO: grab entire verb/noun phrase to the left of the adverb
                        raise NotImplementedError("adverb application to adverb")

                    else:
                        result = apply_adverb_to_verb(verb, adverb)

                    if edge == "(" and last == [")"] and level > 0:
                        return result

                    fragment[1:3] = [result]

                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Noun(), Adverb():
                    edge, noun, adverb = fragment
                    raise NotImplementedError("adverb application to noun not implemented")

                # 4. Conjunction
                case (
                    [None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Verb() | Noun(), Conjunction(), Verb() | Noun()] |
                    [None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Verb() | Noun(), Conjunction(), Verb() | Noun(), _]
                ):
                    edge, verb_or_noun_1, conjunction, verb_or_noun_2, *last = fragment
                    # TODO: find entire verb phrase on the left of the conjunction before applying the conjunction
                    # See: https://code.jsoftware.com/wiki/Vocabulary/Modifiers
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
                    if not isinstance(verb_2, Verb):
                        raise NotImplementedError("Fork currently implemented only for verb/verb/verb")
                    result = build_fork(verb_or_noun_1, verb_2, verb_3)
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 6. Hook/Adverb
                case (
                    [None | "=." | "=:" | "(", Conjunction() | Adverb() | Verb() | Noun(), Conjunction() | Adverb() | Verb() | Noun()]
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
                case Name(), "=." | "=:", Conjunction() | Adverb() | Verb() | Noun(), _:
                    raise NotImplementedError("copula")

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

    return fragment
