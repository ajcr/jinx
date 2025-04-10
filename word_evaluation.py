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
    Atom,
    Array,
)

from np_implementation import (
    array_to_string,
    atom_to_string,
    apply_monad,
    apply_dyad,
)


def str_(word: Atom | Array | Verb) -> str:
    """Print the word in a human-readable format."""
    if isinstance(word, str):
        return word
    if isinstance(word, Atom):
        return atom_to_string(word)
    elif isinstance(word, Array):
        return array_to_string(word)
    elif isinstance(word, Verb):
        return word.spelling
    else:
        raise NotImplementedError(f"Cannot print word of type {type(word)}")


def print_words(words: list[PartOfSpeechT]) -> None:
    """Print the words in a human-readable format."""
    print(" ".join(str_(word) for word in words if word is not None))


def evaluate_words(words: list[PartOfSpeechT]) -> list[PartOfSpeechT]:
    """Evaluate the words in the sentence."""

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

        fragment = [word, *fragment]

        # fmt: off
        while True:
            match fragment:
                # 0. Monad
                case (
                    [None | "=." | "=:" | "(", Verb(), Noun(), _] |
                    [None | "=." | "=:" | "(", Verb(), Noun()]

                ):
                    _, verb, noun = fragment
                    result = apply_monad(verb, noun)
                    fragment[1:] = [result]

                # 1. Monad
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb(),
                    Verb(),
                    Noun(),
                ):
                    _, _, verb, noun = fragment
                    result = apply_monad(verb, noun)
                    fragment[2:] = [result]

                # 2. Dyad
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Noun(),
                    Verb(),
                    Noun(),
                ):
                    _, noun, verb, noun_2 = fragment
                    result = apply_dyad(verb, noun, noun_2)
                    fragment[1:] = [result]

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
                case (
                    ["(", Conjunction() | Adverb() | Verb() | Noun(), ")", _] |
                    ["(", Conjunction() | Adverb() | Verb() | Noun(), ")"]
                ):
                    _, cavn, *_ = fragment
                    fragment = [cavn]

                # Non-executable fragment.
                case _:
                    break

        # fmt: on

    return fragment
