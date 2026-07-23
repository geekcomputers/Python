"""
Random Number Guessing Game
---------------------------
This is a simple multiplayer game where each player tries to guess a number
chosen randomly by the computer between 1 and 100. After each guess, the game
provides feedback whether the guess is higher or lower than the target number.
The winner is the player who guesses the number in the fewest attempts.

Example:
    >>> import builtins, random
    >>> random.seed(0)
    >>> inputs = iter(["1", "Alice", "50", "49"])
    >>> builtins.input = lambda prompt="": next(inputs)
    >>> from game import play_game
    >>> players, scores, winners = play_game()
    >>> players
    ['Alice']
    >>> scores  # doctest: +ELLIPSIS
    [2]
    >>> winners
    ['Alice']
"""

import random
from typing import List, Tuple


def get_players(n: int) -> List[str]:
    """
    Prompt to enter `n` player names.

    Args:
        n (int): number of players

    Returns:
        List[str]: list of player names

    Example:
        >>> import builtins
        >>> inputs = iter(["Alice", "Bob"])
        >>> builtins.input = lambda prompt="": next(inputs)
        >>> get_players(2)
        ['Alice', 'Bob']
    """
    return [input("Enter name of player: ") for _ in range(n)]


def play_turn(player: str) -> int:
    """
    Let a player try to guess a random number.

    Args:
        player (str): player name

    Returns:
        int: number of attempts taken

    Example:
        >>> import builtins, random
        >>> random.seed(1)
        >>> inputs = iter(["30", "15", "9"])
        >>> builtins.input = lambda prompt="": next(inputs)
        >>> play_turn("Alice")  # doctest: +ELLIPSIS
        3
    """
    target = random.randint(1, 100)
    print(f"\n{player}, it's your turn!")
    attempts = 0
    while True:
        guess = int(input("Please enter your guess: "))
        attempts += 1
        if guess > target:
            print("Too high, try smaller...")
        elif guess < target:
            print("Too low, try bigger...")
        else:
            print("Congratulations! You guessed it!")
            return attempts


def play_game() -> Tuple[List[str], List[int], List[str]]:
    """
    Run the multiplayer game.

    Returns:
        Tuple[List[str], List[int], List[str]]: (players, scores, winners)

    Example:
        >>> import builtins, random
        >>> random.seed(2)
        >>> inputs = iter(["1", "Eve", "30", "13"])
        >>> builtins.input = lambda prompt="": next(inputs)
        >>> players, scores, winners = play_game()
        >>> players
        ['Eve']
        >>> scores  # doctest: +ELLIPSIS
        [2]
        >>> winners
        ['Eve']
    """
    n = int(input("Enter number of players: "))
    players = get_players(n)
    scores = [play_turn(p) for p in players]
    min_score = min(scores)
    winners = [p for p, s in zip(players, scores) if s == min_score]
    print("\nResults:")
    for p, s in zip(players, scores):
        print(f"{p}: {s} attempts")
    print("\nWinner(s):", ", ".join(winners))
    return players, scores, winners


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    play_game()
