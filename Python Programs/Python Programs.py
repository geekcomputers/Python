#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Utilities Collection

This module contains a collection of common algorithms and utilities,
including number theory, string manipulation, linked list operations,
and classic puzzles. All functions are optimized and fully type-annotated.

Functions:
    - reverse_number(num)
    - print_table(n)
    - reverse_linked_list_recursive(head)
    - reverse_linked_list_iterative(head)
    - factorial(n)
    - sqrt(num)
    - product_of_unique_prime_factors(n)   # uses sympy for speed
    - tower_of_hanoi(n, source, dest, aux)
    - count_vowels(text)
    - lcm(a, b)
    - fibonacci_recursive(n)
    - fibonacci_sequence(n)
    - mail_merge(names_file, body_file)
    - remove_punctuation(text)
    - sort_words(text)
    - transpose_matrix(matrix)

Dependencies:
    - sympy (for prime factors)
"""

import math
import sys
from typing import Dict, List, Optional

from sympy import primefactors


# ---------- Linked List Node ----------
class Node:
    """Singly linked list node."""

    def __init__(self, data: int) -> None:
        self.data: int = data
        self.next: Optional["Node"] = None


# ---------- 1. Reverse a number ----------
def reverse_number(num: int) -> int:
    """
    Reverse the digits of an integer.

    Args:
        num: A non-negative integer.

    Returns:
        The reversed integer.

    Examples:
        >>> reverse_number(12345)
        54321
        >>> reverse_number(1000)
        1
        >>> reverse_number(0)
        0
    """
    if num < 0:
        raise ValueError("num must be non-negative")
    rev = 0
    while num > 0:
        rev = rev * 10 + num % 10
        num //= 10
    return rev


# ---------- 2. Print multiplication table ----------
def print_table(n: int, upto: int = 10) -> None:
    """
    Print the multiplication table for n from 1 to upto.

    Args:
        n: The number.
        upto: Upper limit (default 10).

    Examples:
        >>> print_table(2, 3)
        2 x 1 = 2
        2 x 2 = 4
        2 x 3 = 6
    """
    for i in range(1, upto + 1):
        print(f"{n} x {i} = {n * i}")


# ---------- 3. Reverse Linked List (Recursive) ----------
def reverse_linked_list_recursive(head: Optional[Node]) -> Optional[Node]:
    """
    Reverse a singly linked list recursively.

    Args:
        head: Head node of the list.

    Returns:
        New head of the reversed list.

    Examples:
        >>> head = Node(1); head.next = Node(2); head.next.next = Node(3)
        >>> new = reverse_linked_list_recursive(head)
        >>> new.data, new.next.data, new.next.next.data
        (3, 2, 1)
    """
    if head is None or head.next is None:
        return head
    new_head = reverse_linked_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head


# ---------- 4. Reverse Linked List (Iterative) ----------
def reverse_linked_list_iterative(head: Optional[Node]) -> Optional[Node]:
    """
    Reverse a singly linked list iteratively.

    Args:
        head: Head node.

    Returns:
        New head of the reversed list.

    Examples:
        >>> head = Node(10); head.next = Node(20)
        >>> new = reverse_linked_list_iterative(head)
        >>> new.data, new.next.data
        (20, 10)
    """
    prev = None
    curr = head
    while curr is not None:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev


# ---------- 5. Factorial ----------
def factorial(n: int) -> int:
    """
    Compute the factorial of n (n!) using an iterative method.

    Args:
        n: Non-negative integer.

    Returns:
        n! (0! = 1).

    Raises:
        ValueError: If n is negative.

    Examples:
        >>> factorial(5)
        120
        >>> factorial(0)
        1
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# ---------- 6. Square Root ----------
def sqrt(num: float) -> float:
    """
    Compute the square root of a non-negative number.

    Args:
        num: Non-negative float or int.

    Returns:
        The square root.

    Raises:
        ValueError: If num is negative.

    Examples:
        >>> sqrt(9.0)
        3.0
        >>> sqrt(2)
        1.4142135623730951
    """
    if num < 0:
        raise ValueError("Cannot compute square root of negative number")
    return math.sqrt(num)


# ---------- 7. Product of unique prime factors (SymPy accelerated) ----------
def product_of_unique_prime_factors(n: int) -> int:
    """
    Compute the product of distinct prime factors of n.

    This function uses SymPy's primefactors for fast computation.

    Args:
        n: Positive integer.

    Returns:
        Product of unique prime factors.

    Raises:
        ValueError: If n < 1.

    Examples:
        >>> product_of_unique_prime_factors(44)
        22
        >>> product_of_unique_prime_factors(12)
        6
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    prod = 1
    for p in primefactors(n):
        prod *= p
    return prod


# ---------- 8. Tower of Hanoi ----------
def tower_of_hanoi(n: int, source: str, dest: str, aux: str) -> None:
    """
    Solve the Tower of Hanoi puzzle and print the moves.

    Args:
        n: Number of disks.
        source: Name of source peg.
        dest: Name of destination peg.
        aux: Name of auxiliary peg.

    Examples:
        >>> tower_of_hanoi(1, 'A', 'B', 'C')
        Move disk 1 from source A to destination B
    """
    if n == 1:
        print(f"Move disk 1 from source {source} to destination {dest}")
        return
    tower_of_hanoi(n - 1, source, aux, dest)
    print(f"Move disk {n} from source {source} to destination {dest}")
    tower_of_hanoi(n - 1, aux, dest, source)


# ---------- 9. Count vowels ----------
def count_vowels(text: str) -> Dict[str, int]:
    """
    Count the occurrences of each vowel (a, e, i, o, u) in a string.

    Args:
        text: Input string.

    Returns:
        Dictionary with vowels as keys and counts as values.

    Examples:
        >>> count_vowels("Hello World")
        {'a': 0, 'e': 1, 'i': 0, 'o': 2, 'u': 0}
    """
    vowels = "aeiou"
    text_lower = text.lower()
    count = {v: 0 for v in vowels}
    for ch in text_lower:
        if ch in count:
            count[ch] += 1
    return count


# ---------- 10. Least Common Multiple (LCM) ----------
def lcm(a: int, b: int) -> int:
    """
    Compute the least common multiple of two positive integers.

    Args:
        a, b: Positive integers.

    Returns:
        LCM of a and b.

    Raises:
        ValueError: If either is <= 0.

    Examples:
        >>> lcm(54, 24)
        216
        >>> lcm(7, 3)
        21
    """
    if a <= 0 or b <= 0:
        raise ValueError("Both arguments must be positive")
    return abs(a * b) // math.gcd(a, b)


# ---------- 11. Fibonacci (Recursive) ----------
def fibonacci_recursive(n: int) -> int:
    """
    Return the n-th Fibonacci number (0-indexed) recursively.

    Args:
        n: Non-negative integer.

    Returns:
        Fibonacci number F(n) with F(0)=0, F(1)=1.

    Raises:
        ValueError: If n < 0.

    Examples:
        >>> fibonacci_recursive(0)
        0
        >>> fibonacci_recursive(6)
        8
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


# ---------- 12. Fibonacci Sequence (Iterative) ----------
def fibonacci_sequence(n: int) -> List[int]:
    """
    Generate the first n Fibonacci numbers (starting from F(0)=0).

    Args:
        n: Number of terms (n >= 0).

    Returns:
        List of the first n Fibonacci numbers.

    Raises:
        ValueError: If n < 0.

    Examples:
        >>> fibonacci_sequence(5)
        [0, 1, 1, 2, 3]
        >>> fibonacci_sequence(1)
        [0]
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return []
    seq = [0] * n
    if n > 1:
        seq[1] = 1
    for i in range(2, n):
        seq[i] = seq[i - 1] + seq[i - 2]
    return seq


# ---------- 13. Mail Merger ----------
def mail_merge(names_file: str, body_file: str, output_prefix: str = "mail_") -> None:
    """
    Merge names from a file with a mail body template and write individual mail files.

    Args:
        names_file: Path to file with one name per line.
        body_file: Path to file containing the mail body (with placeholders).
        output_prefix: Prefix for output filenames.

    The body file can contain a placeholder like {name} which will be replaced.
    If no placeholder, the name is prepended as "Hello <name>" line.

    Examples:
        Assuming names.txt contains "Alice\\nBob" and body.txt contains "Welcome!",
        this will create mail_Alice.txt and mail_Bob.txt.
    """
    try:
        with open(names_file, "r", encoding="utf-8") as nf:
            names = [line.strip() for line in nf if line.strip()]
        with open(body_file, "r", encoding="utf-8") as bf:
            body_template = bf.read()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return

    for name in names:
        mail_body = body_template.replace("{name}", name)
        if "{name}" not in body_template:
            mail_body = f"Hello {name}\n{body_template}"
        with open(f"{output_prefix}{name}.txt", "w", encoding="utf-8") as mf:
            mf.write(mail_body)


# ---------- 14. Remove Punctuation ----------
def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation characters from a string.

    Punctuation defined as: !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~

    Args:
        text: Input string.

    Returns:
        String without punctuation.

    Examples:
        >>> remove_punctuation("Hello!!!, he said ---and went.")
        'Hello he said and went'
    """
    import string

    return "".join(ch for ch in text if ch not in string.punctuation)


# ---------- 15. Sort Words Alphabetically ----------
def sort_words(text: str) -> Dict[int, str]:
    """
    Sort all words in a string alphabetically, remove duplicates, and return
    a dictionary mapping sequential numbers to each unique word.

    Args:
        text: Input string (punctuation removed automatically).

    Returns:
        Dictionary {1: first_word, 2: second_word, ...}.

    Examples:
        >>> sort_words("Hello world hello")
        {1: 'hello', 2: 'world'}
    """
    cleaned = remove_punctuation(text).lower()
    words = cleaned.split()
    seen = set()
    unique_sorted = []
    for word in sorted(words):
        if word not in seen:
            seen.add(word)
            unique_sorted.append(word)
    return {i + 1: word for i, word in enumerate(unique_sorted)}


# ---------- 16. Transpose Matrix ----------
def transpose_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Transpose a 2D matrix (list of lists).

    Args:
        matrix: A rectangular matrix.

    Returns:
        Transposed matrix.

    Examples:
        >>> transpose_matrix([[12, 7], [4, 5], [3, 8]])
        [[12, 4, 3], [7, 5, 8]]
    """
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]


# ---------- Demo / Test ----------
if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
