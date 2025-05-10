# Jinx

A work-in-progress J interpreter written in Python and NumPy.

## Examples

Start the interactive shell with `jinx`. As in the official J implementation, the shell prompt is four spaces, so commands appear indented.

Atoms (scalars) and arrays with 1 or more dimensions (rank):
```j
    3           NB. single integer (atom)
3

    3 5 7 11 13  NB. rank 1 array of integers (atoms)
3 5 7 11 13

    3 2 $ 7 1 0  NB. rank 2 array created using $ dyad (values repeated to fill shape)
7 1
0 7
1 0
```

Monadic and dyadic application of verbs (applied right-to-left, parentheses force precedence):
```j
     -3.141            # NB. monadic -
_3.141

    10 - 3 5 7 11 13   # NB. dyadic -
7 5 3 _1 _3

    8 % 4 - 2    NB. dyad % is division, but the subtraction with - is done first
4

    (8 % 4) - 2  NB. force division before substraction
0
```

Verbs can apply to nouns of different rank. The rank of a verb can be modified (creating a new verb), changing how it applies over its noun arguments:
```j
    (2 2 $ 10 100 1000 10000) + (i. 2 2 2)      NB. default rank of + is (0 0)
   10    11
  102   103

 1004  1005
10006 10007

    (2 2 $ 10 100 1000 10000) +"0 2 (i. 2 2 2)  NB. conjunction " changes rank of + to (0 2)
   10    11
   12    13

  100   101
  102   103


 1004  1005
 1006  1007

10004 10005
10006 10007
```

Adverbs modify verbs, e.g. `/` inserts the verb between items of its argument:
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

This project is primarily a learning exercise: I want to improve my patchy understanding of J by implementing a useful subset of the language and its core concepts (rank, modifiers, etc.).

It is also an attempt to prototype an interpreter for an array language that can be executed using Python's different array and tensor frameworks (NumPy, JAX, PyTorch, etc.) according to user's choice.