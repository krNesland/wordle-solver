"""
Created 11/02/2022
Kristoffer Nesland, kristoffer@solutionseeker.no

Playing around with Python classes.

# TODO: Should have a look-up of Word1 * Word2 = Pattern (over 100M, so would require a lot of memory).
"""

from __future__ import annotations

import copy
import itertools as itt
import math
import random
import typing as ty
from functools import partial
from multiprocessing import Pool

import numpy as np
import plotly.graph_objects as go

from data import ALL, SECRET

random.seed(int.from_bytes(b"solsee", byteorder="little"))


WORD_LEN: int = 5


class Color:
    gray = 0
    yellow = 1
    green = 2


class Pattern:
    """
    E.g. yellow, gray, gray, green, yellow
    """
    def __init__(self, colors: ty.List[int]):
        assert len(colors) == WORD_LEN, f"Pattern must have a length of {WORD_LEN}, got {len(colors)}"
        for c in colors:
            assert c in [Color.gray, Color.yellow, Color.green], f"Entries in pattern must be in [{Color.gray}, {Color.yellow}, {Color.green}], got {c}"

        self._colors: ty.Tuple[int] = tuple(colors)

    def __str__(self) -> str:
        return "".join(map(str, self._colors))

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key: int):
        return self._colors[key]

    def __eq__(self, other) -> bool:
        return self._colors == other._colors

    def __hash__(self) -> int:
        return int(str(self))

    @classmethod
    def get_all(cls: Pattern) -> ty.List[Pattern]:
        patterns: ty.List[Pattern] = list()

        looper = itt.product(*[
            [Color.gray, Color.yellow, Color.green]
            for i in range(WORD_LEN)
        ])

        for i in looper:
            patterns.append(cls(list(i)))

        return patterns

    def count_matches(self, patterns: ty.List[Pattern]) -> int:
        """
        Number of matching patterns.
        """
        num: int = 0
        for pattern in patterns:
            if self == pattern:
                num += 1

        return num


class Word:
    """
    E.g. snack
    """
    def __init__(self, word: str):
        word = word.lower()

        assert len(word) == WORD_LEN, f"Word must be of length {WORD_LEN}, got {len(word)}"
        assert word.isalpha(), "Word must be all numeric"
        assert word in ALL, "Word does not exist"
        self._word: str = word

    def __getitem__(self, key: int):
        return self._word[key]

    def __repr__(self) -> str:
        return self._word

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < WORD_LEN:
            letter = self._word[self.i]
            self.i += 1
            return letter
        else:
            raise StopIteration

    def __eq__(self, other: Word) -> bool:
        return self._word == other._word

    def __mul__(self, other: Word) -> Pattern:
        """
        Pattern of self compared to other.
        """
        colors: ty.List[int] = list()
        other_rem = other._word

        for i, (l1, l2) in enumerate(zip(self, other)):
            if l1 == l2:
                colors.append(Color.green)
                other_rem = other_rem.replace(l1, "", 1)
            elif l1 in other_rem:
                colors.append(Color.yellow)
                other_rem = other_rem.replace(l1, "", 1)
            else:
                colors.append(Color.gray)

        return Pattern(colors)

    def __hash__(self) -> int:
        # Perhaps not bullet proof?
        return hash(self._word)

    @classmethod
    def get_all(cls: Word) -> ty.List[Word]:
        return [
            cls(word)
            for word in ALL
        ]


def get_matches(
        guess: Word,
        remaining_words: ty.List[Word],
        pattern: Pattern,
) -> ty.List[Word]:
    matching_words: ty.List[Word] = list()

    for word in remaining_words:
        this_pattern = guess * word
        if this_pattern == pattern:
            matching_words.append(word)

    return matching_words


def generate_patterns(guess: Word, words: ty.List[Word]) -> ty.List[Pattern]:
    """
    Generate the corresponding pattern for each word.
    """
    return [
        guess * word
        for word in words
    ]


def get_probabilities(
        generated_patterns: ty.List[Pattern],
        all_patterns: ty.List[Pattern],
) -> ty.Dict[Pattern, float]:
    matches: ty.Dict[Pattern, int] = dict()
    for pattern in all_patterns:
        matches[pattern] = pattern.count_matches(generated_patterns)

    return {
        pattern: matches[pattern] / len(generated_patterns)
        for pattern in all_patterns
        if matches[pattern] != 0
    }


def calc_expected_entropy(probabilities: ty.Dict[Pattern, float]) -> float:
    return sum(map(lambda x: x * math.log2(1 / x), probabilities.values()))


def get_top_guess(
        remaining_words: ty.List[Word],
        all_words: ty.List[Word],
        all_patterns: ty.List[Pattern],
) -> Word:
    return get_top_guesses(remaining_words, all_words, all_patterns)[0]


def get_expected_entropy(word: Word, remaining_words: ty.List[Word], all_patterns: ty.List[Pattern]) -> float:
    generated_patterns = generate_patterns(word, remaining_words)
    probabilities = get_probabilities(generated_patterns, all_patterns)
    return calc_expected_entropy(probabilities)


def get_top_guesses(
        remaining_words: ty.List[Word],
        all_words: ty.List[Word],
        all_patterns: ty.List[Pattern],
        num: int = 10,
        verbose: bool = False,
) -> ty.List[Word]:
    if verbose:
        print("Generating top guesses...")

    # Trying out some fancy pants parallelization.
    get_expected_entropy_mod = partial(get_expected_entropy, remaining_words=remaining_words, all_patterns=all_patterns)
    with Pool(processes=6) as p:
        expected_entropies = p.map(get_expected_entropy_mod, all_words)

    idxs = list(np.flip(np.argsort(expected_entropies)))
    top_guesses: ty.List[Word] = list()

    for i in range(num):
        idx = idxs[i]
        top_guesses.append(all_words[idx])
        if verbose:
            print(i, all_words[idx], f"{expected_entropies[idx]:.3f}")

    return top_guesses


def plot_probabilities(probabilities: ty.Dict[Pattern, float]):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(probabilities))),
        y=list(probabilities.values()),
        hovertext=list(map(str, probabilities.keys())),
    ))
    fig.update_layout(
        bargap=0.0,
    )
    fig.show()


def guess_loop(remaining_words: ty.List[Word]) -> ty.List[Word]:
    guess: ty.Optional[Word] = None
    print("")
    while guess is None:
        try:
            guess = Word(input("Guess: "))
        except AssertionError as e:
            print(e)

    if SECRET is not None:
        pattern: Pattern = guess * Word(SECRET)
        print(f"Pattern: {pattern}")
    else:
        pattern: ty.Optional[Pattern] = None
        while pattern is None:
            try:
                pattern = Pattern([
                    int(digit)
                    for digit in input("Pattern: ")
                ])
            except AssertionError as e:
                print(e)
            except ValueError:
                print("Input must be integers")

    num_before: int = len(remaining_words)
    remaining_words = get_matches(guess, remaining_words, pattern)
    num_after: int = len(remaining_words)
    if num_after == 0:
        raise ValueError("No words remaining")
    print(f"Entropy gotten: {math.log2((num_before + 0.0) / num_after):.3f}")
    print(f"#Remaining words: {len(remaining_words)}")
    print("Some of the remaining words:")
    for i, word in enumerate(remaining_words):
        print(word)
        if i >= 10:
            break

    return remaining_words


def main():
    print("Welcome to Wordle!")
    if SECRET is None:
        print("Secret is not set and you must manually enter patterns.")
    else:
        print("Secret is set and pattern will automatically be calculated.")

    all_patterns = Pattern.get_all()
    all_words = Word.get_all()
    remaining_words = Word.get_all()

    remaining_words = guess_loop(remaining_words)

    while len(remaining_words) > 1:
        get_top_guesses(remaining_words, all_words, all_patterns, verbose=True)
        remaining_words = guess_loop(remaining_words)

    print(f"Only remaining: {remaining_words[0]}")


if __name__ == "__main__":
    main()
