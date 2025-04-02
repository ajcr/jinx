import cmd

from word_formation import form_words
from word_spelling import spell_words
from word_evaluation import evaluate_words


class Shell(cmd.Cmd):
    prompt = "(jinx) "

    def do_exit(self, _):
        return True

    def default(self, line):
        words = form_words(line)
        words = spell_words(words)
        words = evaluate_words(words)
        print(words)


if __name__ == "__main__":
    Shell().cmdloop()