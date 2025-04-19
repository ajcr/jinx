# Jinx

J interpreter written in Python, backed with NumPy for execution (with scope to execute with JAX, PyTorch, or another library in future).

A work in progress.

## Examples

Start the interactive shell with `jinx`. As in the official J implementation, the shell prompt is four spaces, so commands appear indented.

Atoms (scalars) and arrays with 1 or more dimensions (rank):
```j
    _3           NB. single integer (atom)
_3

    3 5 7 11 13  NB. rank 1 array of integers (atoms)
3 5 7 11 13

    3 2 $ 7 1 0  NB. rank 2 array created using $ dyad, values repeated to fill shape
7 1
0 7
1 0
```

Monadic and dyadic application of verbs (applied right-to-left, parentheses force precedence):
```j
     -3.141
_3.141

    10 - 3 5 7 11 13
7 5 3 _1 _3

    8 % 4 - 2    NB. dyad % is division, but subtraction done first
4

    (8 % 4) - 2  NB. force division before substraction
0
```

Adverbs to modify verbs (e.g. change the rank that a verb will apply with):
```j
    i. 2 3 4       NB. a rank 3 array created with (i.)
 0  1  2  3
 4  5  6  7
 8  9 10 11

12 13 14 15
16 17 18 19
20 21 22 23

    +/ i. 2 3 4    NB. sum (infinite rank) applies to array as rank 3, the default
12 14 16 18
20 22 24 26
28 30 32 34

    +/"2 i. 2 3 4  NB. sum applies as rank 2
12 15 18 21
48 51 54 57

    +/"1 i. 2 3 4  NB. sum applies as rank 1
 6 22 38
54 70 86
```

## Motivation

This project is primarily a learning exercise: I want to improve my patchy understanding of J by implementing a useful subset of the language and its core concepts (ranks, obverses, etc.).

It is also an attempt to prototype an interpreter for an array language that can be executed using different array frameworks (NumPy, JAX, PyTorch, etc.) according to the user's choice.