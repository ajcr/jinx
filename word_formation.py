"""Word Formation.

Given a sentence (a string of characters), form a list of its constituent words.

Based on the description from 'An Implementation of J': https://www.jsoftware.com/ioj/iojSent.htm).

See also: https://code.jsoftware.com/wiki/Vocabulary/Words#WordFormation

The terse naming of states and character classes has been preserved for the sake of consistency
with the original description.

"""

from enum import StrEnum
from typing import Mapping

from vocabulary import Word


class State(StrEnum):
    S = "space"
    X = "other"
    A = "alphanumeric"
    N = "N"
    NB = "NB"
    NINE = "numeric"
    Q = "quote"
    QQ = "even quotes"
    Z = "trailing comment"


class CharacterClass(StrEnum):
    S = "space"
    X = "other"
    A = "letters excl. NB"
    N = "N"
    B = "B"
    NINE = "digits and _"
    D = "."
    C = ":"
    Q = "'"


class Action(StrEnum):
    I = "emit, update"
    N = "no emit, update"
    X = "no action"


STATE_TRANSITION: Mapping[tuple[State, CharacterClass], tuple[State, Action]] = {
    # "space"
    (State.S, CharacterClass.X): (State.X, Action.N),
    (State.S, CharacterClass.S): (State.S, Action.X),
    (State.S, CharacterClass.A): (State.A, Action.N),
    (State.S, CharacterClass.N): (State.N, Action.N),
    (State.S, CharacterClass.B): (State.A, Action.N),
    (State.S, CharacterClass.NINE): (State.NINE, Action.N),
    (State.S, CharacterClass.D): (State.X, Action.N),
    (State.S, CharacterClass.C): (State.X, Action.N),
    (State.S, CharacterClass.Q): (State.Q, Action.N),
    # "other"
    (State.X, CharacterClass.X): (State.X, Action.I),
    (State.X, CharacterClass.S): (State.S, Action.I),
    (State.X, CharacterClass.A): (State.A, Action.I),
    (State.X, CharacterClass.N): (State.N, Action.I),
    (State.X, CharacterClass.B): (State.A, Action.I),
    (State.X, CharacterClass.NINE): (State.NINE, Action.I),
    (State.X, CharacterClass.D): (State.X, Action.X),
    (State.X, CharacterClass.C): (State.X, Action.X),
    (State.X, CharacterClass.Q): (State.Q, Action.I),
    # "alphanumeric"
    (State.A, CharacterClass.X): (State.X, Action.I),
    (State.A, CharacterClass.S): (State.S, Action.I),
    (State.A, CharacterClass.A): (State.A, Action.X),
    (State.A, CharacterClass.N): (State.A, Action.X),
    (State.A, CharacterClass.B): (State.A, Action.X),
    (State.A, CharacterClass.NINE): (State.A, Action.X),
    (State.A, CharacterClass.D): (State.X, Action.X),
    (State.A, CharacterClass.C): (State.X, Action.X),
    (State.A, CharacterClass.Q): (State.Q, Action.I),
    # Action.N
    (State.N, CharacterClass.X): (State.X, Action.I),
    (State.N, CharacterClass.S): (State.S, Action.I),
    (State.N, CharacterClass.A): (State.A, Action.X),
    (State.N, CharacterClass.N): (State.A, Action.X),
    (State.N, CharacterClass.B): (State.NB, Action.X),
    (State.N, CharacterClass.NINE): (State.A, Action.X),
    (State.N, CharacterClass.D): (State.X, Action.X),
    (State.N, CharacterClass.C): (State.X, Action.X),
    (State.N, CharacterClass.Q): (State.Q, Action.I),
    # "NB"
    (State.NB, CharacterClass.X): (State.X, Action.I),
    (State.NB, CharacterClass.S): (State.S, Action.I),
    (State.NB, CharacterClass.A): (State.A, Action.X),
    (State.NB, CharacterClass.N): (State.A, Action.X),
    (State.NB, CharacterClass.B): (State.A, Action.X),
    (State.NB, CharacterClass.NINE): (State.A, Action.X),
    (State.NB, CharacterClass.D): (State.Z, Action.X),
    (State.NB, CharacterClass.C): (State.X, Action.X),
    (State.NB, CharacterClass.Q): (State.Q, Action.I),
    # "Z"
    (State.Z, CharacterClass.X): (State.Z, Action.X),
    (State.Z, CharacterClass.S): (State.Z, Action.X),
    (State.Z, CharacterClass.A): (State.Z, Action.X),
    (State.Z, CharacterClass.N): (State.Z, Action.X),
    (State.Z, CharacterClass.B): (State.Z, Action.X),
    (State.Z, CharacterClass.NINE): (State.Z, Action.X),
    (State.Z, CharacterClass.D): (State.X, Action.X),
    (State.Z, CharacterClass.C): (State.X, Action.X),
    (State.Z, CharacterClass.Q): (State.Z, Action.X),
    # "NINE"
    (State.NINE, CharacterClass.X): (State.X, Action.I),
    (State.NINE, CharacterClass.S): (State.S, Action.I),
    (State.NINE, CharacterClass.A): (State.NINE, Action.X),
    (State.NINE, CharacterClass.N): (State.NINE, Action.X),
    (State.NINE, CharacterClass.B): (State.NINE, Action.X),
    (State.NINE, CharacterClass.NINE): (State.NINE, Action.X),
    (State.NINE, CharacterClass.D): (State.NINE, Action.X),
    (State.NINE, CharacterClass.C): (State.X, Action.X),
    (State.NINE, CharacterClass.Q): (State.Q, Action.I),
    # "Q"
    (State.Q, CharacterClass.X): (State.Q, Action.X),
    (State.Q, CharacterClass.S): (State.Q, Action.X),
    (State.Q, CharacterClass.A): (State.Q, Action.X),
    (State.Q, CharacterClass.N): (State.Q, Action.X),
    (State.Q, CharacterClass.B): (State.Q, Action.X),
    (State.Q, CharacterClass.NINE): (State.Q, Action.X),
    (State.Q, CharacterClass.D): (State.Q, Action.X),
    (State.Q, CharacterClass.C): (State.Q, Action.X),
    (State.Q, CharacterClass.Q): (State.QQ, Action.X),
    # "QQ"
    (State.QQ, CharacterClass.X): (State.X, Action.I),
    (State.QQ, CharacterClass.S): (State.S, Action.I),
    (State.QQ, CharacterClass.A): (State.A, Action.I),
    (State.QQ, CharacterClass.N): (State.N, Action.I),
    (State.QQ, CharacterClass.B): (State.A, Action.I),
    (State.QQ, CharacterClass.NINE): (State.NINE, Action.I),
    (State.QQ, CharacterClass.D): (State.X, Action.I),
    (State.QQ, CharacterClass.C): (State.X, Action.I),
    (State.QQ, CharacterClass.Q): (State.Q, Action.X),
    # "Z"
    (State.Z, CharacterClass.X): (State.Z, Action.X),
    (State.Z, CharacterClass.S): (State.Z, Action.X),
    (State.Z, CharacterClass.A): (State.Z, Action.X),
    (State.Z, CharacterClass.N): (State.Z, Action.X),
    (State.Z, CharacterClass.B): (State.Z, Action.X),
    (State.Z, CharacterClass.NINE): (State.Z, Action.X),
    (State.Z, CharacterClass.D): (State.Z, Action.X),
    (State.Z, CharacterClass.C): (State.Z, Action.X),
    (State.Z, CharacterClass.Q): (State.Z, Action.X),
}


def get_character_class(char: str) -> CharacterClass:
    if char.isspace():
        return CharacterClass.S

    if char == "N":
        return CharacterClass.N

    if char == "B":
        return CharacterClass.B

    if char.isalpha():
        return CharacterClass.A

    if char.isdigit() or char == "_":
        return CharacterClass.NINE

    if char == ".":
        return CharacterClass.D

    if char == ":":
        return CharacterClass.C

    if char == "'":
        return CharacterClass.Q

    return CharacterClass.X


def form_words(sentence: str) -> list[Word]:
    if len(sentence) == 0:
        return []

    # Append whitespace EOS marker to ensure that the final word is emitted.
    # This has the side effect that comments and unterminated quotes will not
    # be emitted, so we handle this later in the function.
    sentence += "\n"

    i = j = 0
    current_state: State = State.S

    words: list[Word] = []

    # A sequence of numbers separated by whitespace is treated as a single word in J.
    # Set a flag to handle this when the current numeric word needs to be emitted.
    continue_numeric = False

    while i < len(sentence):
        char = sentence[i]
        char_class = get_character_class(char)
        new_state, action = STATE_TRANSITION[(current_state, char_class)]

        if (
            words
            and words[-1].is_numeric
            and current_state == State.S
            and new_state == State.NINE
        ):
            continue_numeric = True

        if action == Action.I:
            if continue_numeric:
                prev_word = words.pop()
                value = sentence[prev_word.start : i]
                word = Word(value=value, is_numeric=True, start=prev_word.start, end=i)
                words.append(word)
                continue_numeric = False

            else:
                value = sentence[j:i]
                is_numeric = current_state == State.NINE
                word = Word(value=value, is_numeric=is_numeric, start=j, end=i)
                words.append(word)

            j = i

        elif action == Action.N:
            j = i

        current_state = new_state
        i += 1

    remaining_word = sentence[j : i - 1]  # i-1 to exclude EOS marker.
    if remaining_word and not remaining_word.isspace():
        words.append(Word(value=remaining_word, is_numeric=False, start=j, end=i - 1))

    return words
