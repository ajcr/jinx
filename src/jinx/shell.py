import argparse
import cmd
import sys

from jinx.errors import BaseJError, SpellingError
from jinx.execution.executor import Executor, load_executor
from jinx.word_evaluation import evaluate_words, print_words
from jinx.word_formation import form_words
from jinx.word_spelling import spell_words


class Shell(cmd.Cmd):
    prompt = "    "

    def __init__(self, executor: Executor):
        super().__init__()
        self.variables = {}
        self.executor = executor

    def do_exit(self, _):
        return True

    def default(self, line):
        words = form_words(line)
        try:
            words = spell_words(words)
        except SpellingError as e:
            print(e, file=sys.stderr)
            return
        try:
            words = evaluate_words(self.executor, words, self.variables)
            print_words(self.executor, words, self.variables)
        except BaseJError as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)

    def do_EOF(self, _):
        return True

    # '?' is a primitive verb in J and we want the Cmd class to disregard it.
    # and not treat it as a help command.
    def do_help(self, line):
        return self.default("?" + line)


def main():
    """Entry point for the jinx shell."""
    parser = argparse.ArgumentParser(description="Jinx shell")
    parser.add_argument(
        "--executor",
        type=str,
        choices=["numpy"],
        default="numpy",
        help="Executor to use.",
    )
    args = parser.parse_args()

    # Currently only the "numpy" executor is implemented.
    executor = load_executor(args.executor)

    try:
        Shell(executor).cmdloop()
    except EOFError:
        pass


if __name__ == "__main__":
    main()
