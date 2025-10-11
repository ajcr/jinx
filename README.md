# Jinx

![ci](https://github.com/ajcr/jinx/actions/workflows/ci.yaml/badge.svg?branch=main)

An experimental interpreter for the J programming language, built on top of [NumPy](https://numpy.org/).

Implements many of J's primitives and tacit programming capabilities, and can be extended to support execution via other frameworks too.

## Executing J

Start the interactive shell:
```sh
jinx
```
The shell prompt is four spaces, so commands appear indented. Internally, all multidimensional arrays are NumPy arrays. Verbs, conjunctions and adverbs are a mixture of Python and NumPy methods.

Here are some examples what Jinx can do so far:

- Solve the "trapping rainwater" problem (solution taken from [here](https://mmapped.blog/posts/04-square-joy-trapped-rain-water)):
```j
    +/@((>./\ <. >./\.)-]) 0 1 0 2 1 0 1 3 2 1 2 1
6
```
- Compute the correlation between two arrays of numbers (taken from [here](https://stackoverflow.com/a/44845495/3923281)). This is a complex combination of different verbs, adverbs, conjunctions and trains:
```j
    2 1 1 7 9 (+/@:* % *&(+/)&.:*:)&(- +/%#) 6 3 1 5 7
0.721332
```
- Create identity matrices in inventive ways (see [this essay](https://code.jsoftware.com/wiki/Essays/Identity_Matrix)):
```j
    |.@~:\ @ ($&0) 3
1 0 0
0 1 0
0 0 1

    (i.@,~ = >: * i.) 3
1 0 0
0 1 0
0 0 1

    ((={:)\ @ i.) 3
1 0 0
0 1 0
0 0 1
```
- Solve the Josephus problem (see [this essay](https://code.jsoftware.com/wiki/Essays/Josephus_Problem)). Calculate the survivor's number for a circle of people of size N. Note the use of verb obverse and the rank conjunction:
```j
    (1&|.&.#:)"0 >: i. 5 10    NB. N ranges from 1 to 50 here (arranged as a table)
 1  1  3  1  3  5  7  1  3  5
 7  9 11 13 15  1  3  5  7  9
11 13 15 17 19 21 23 25 27 29
31  1  3  5  7  9 11 13 15 17
19 21 23 25 27 29 31 33 35 37
```
- Build nested boxes containing heterogeneous data types and print the contents:
```j
    (<<'abc'),(<(<'de',.'fg'),(<<i. 5 2)),(<(<"0 ] % i. 2 2 3))
┌─────┬──────────┬────────────────────────────┐
│┌───┐│┌──┬─────┐│┌────────┬────────┬────────┐│
││abc│││df│┌───┐│││_       │1       │0.5     ││
│└───┘││eg││0 1│││├────────┼────────┼────────┤│
│     ││  ││2 3││││0.333333│0.25    │0.2     ││
│     ││  ││4 5│││└────────┴────────┴────────┘│
│     ││  ││6 7│││                            │
│     ││  ││8 9│││┌────────┬────────┬────────┐│
│     ││  │└───┘│││0.166667│0.142857│0.125   ││
│     │└──┴─────┘│├────────┼────────┼────────┤│
│     │          ││0.111111│0.1     │0.090909││
│     │          │└────────┴────────┴────────┘│
└─────┴──────────┴────────────────────────────┘
```

## Easily Customisable

Everything is in Python. Adding new primitives is easy.

Update the `primitives.py` file with your new part of speech (e.g. a new verb such as `+::`). Write your implementation of this new part of speech in the relevant executor module (e.g. `verbs.py`) and then update the name-to-method mapping at the foot of that module. That's all that's needed.

## Alternative Executors

Execution of sentences is backed by NumPy by default.

However Jinx is designed so that it's possible to implement the primitives using alternative frameworks too. Python many Machine Learning and Scientific Programming libraries that could be used to execution J code.

To prove this concept, there's _highly experimental and incomplete_ support for [JAX](https://docs.jax.dev/en/latest/index.html):
```sh
jinx --executor jax
```
Primitive verbs are JIT compiled and execute on JAX arrays:
```j
    mean =: +/ % #
    mean 33 55 77 100 101
73.2
```

## Warning

This project is experimental. There will be bugs, missing features and performance quirks.

Many key parts of J are not currently implemented (but might be in future). These include:
- Differences in how names are interpreted and resolved at execution time.
- Locales.
- Definitions and direct definitions (using `{{ ... }}`).
- Array types other than floats, integers and and strings.
- Executing J scripts.
- Control words.
