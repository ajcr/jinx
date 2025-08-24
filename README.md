# Jinx

A work-in-progress interpreter for the J programming language, built on top of NumPy.

Supports many features that are central to J, including:
- Multidimensional arrays of integers and floats.
- Many primitive verbs (e.g. `+`, `%:`, `,`, ...), adverbs (`/`, `~`, ...) and conjunctions (`"`, `@:`, ...).
- Correct monadic and dyadic application of verbs of different ranks.
- Obverses.
- Trains (hooks and forks).

## Examples

Start the interactive shell with `jinx`. The shell prompt is four spaces, so commands appear indented.

- The "trapping rainwater" problem (solution from [here](https://mmapped.blog/posts/04-square-joy-trapped-rain-water)):
```j
    +/@((>./\ <. >./\.)-]) 0 1 0 2 1 0 1 3 2 1 2 1
6
```
- The correlation between two arrays of numbers (taken from [here](https://stackoverflow.com/a/44845495/3923281)):
```j
    2 1 1 7 9 (+/@:* % *&(+/)&.:*:)&(- +/%#) 6 3 1 5 7
0.721332
```
- One of many ways to create identity matrices (see [this essay](https://code.jsoftware.com/wiki/Essays/Identity_Matrix)):
```j
    |.@~:\ @ ($&0) 2 3   NB. 3D array with two 3x3 identity matrices
1 0 0
0 1 0
0 0 1

1 0 0
0 1 0
0 0 1
```
- For the Joesphus problem (see [this essay](https://code.jsoftware.com/wiki/Essays/Josephus_Problem)), calculate the survivor's number for a circle of people of size N:
```j
    (1&|.&.#:)"0 >: i. 5 10    NB. N ranges from 1 to 50 here (arranged as a table)
 1  1  3  1  3  5  7  1  3  5
 7  9 11 13 15  1  3  5  7  9
11 13 15 17 19 21 23 25 27 29
31  1  3  5  7  9 11 13 15 17
19 21 23 25 27 29 31 33 35 37
```
- Building and printing nested boxes containing heterogenous datatypes:
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

## Warning

This project is an ongoing learning exercise. There will be bugs, missing features and performance quirks.

Key parts of J not yet implemented in Jinx yet, but might be in future. These include:
- Locales.
- Array types other than floats, integers and and strings.
- Numerous primitives (verbs, conjunctions).
- Control words.
