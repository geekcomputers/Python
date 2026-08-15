# -*- coding: utf-8 -*-
"""
Prime Number Utilities – A comprehensive library for prime‑related operations.

This module is accelerated using the SymPy library for heavy computations
(isprime, factorint, primerange, prime). All functions maintain the same
interface as the original version, with improved performance and full type hints.

Examples:
    >>> isPrime(7)
    True
    >>> getPrime(3)   # 0‑based: 2,3,5,7...
    7
    >>> primeFactorization(12)
    [2, 2, 3]
"""

import math
from functools import lru_cache

from sympy import factorint, isprime
from sympy import prime as sympy_prime
from sympy import primerange

# ---------- Basic utilities ----------


def isEven(number: int) -> bool:
    """Return True if `number` is even, False otherwise.

    Examples:
        >>> isEven(0)
        True
        >>> isEven(1)
        False
    """
    return number % 2 == 0


def isOdd(number: int) -> bool:
    """Return True if `number` is odd, False otherwise.

    Examples:
        >>> isOdd(0)
        False
        >>> isOdd(1)
        True
    """
    return number % 2 != 0


# ---------- Primality testing (accelerated by sympy) ----------


def isPrime(number: int) -> bool:
    """
    Test if `number` is a prime number using SymPy's deterministic isprime.

    Args:
        number: Non‑negative integer.

    Returns:
        bool: True if prime, False otherwise.

    Raises:
        ValueError: If `number` is negative.

    Examples:
        >>> isPrime(0)
        False
        >>> isPrime(1)
        False
        >>> isPrime(2)
        True
        >>> isPrime(97)
        True
        >>> isPrime(10**12 + 39)   # known prime, fast
        True
    """
    if number < 0:
        raise ValueError("number must be non‑negative")
    return isprime(number)


# ---------- Prime number generation (accelerated) ----------


def sieveEr(N: int) -> list[int]:
    """
    Return a list of all primes ≤ N using SymPy's primerange.

    Args:
        N: Upper bound, must be ≥ 2.

    Returns:
        list[int]: Prime numbers from 2 to N (inclusive).

    Raises:
        ValueError: If N < 2.

    Examples:
        >>> sieveEr(10)
        [2, 3, 5, 7]
        >>> sieveEr(2)
        [2]
    """
    if N < 2:
        raise ValueError("N must be ≥ 2")
    return list(primerange(2, N + 1))


def getPrimeNumbers(N: int) -> list[int]:
    """Alias for `sieveEr`."""
    return sieveEr(N)


@lru_cache(maxsize=None)
def getPrime(n: int) -> int:
    """
    Return the n‑th prime number (0‑based indexing) using SymPy's prime().

    Args:
        n: Non‑negative integer.

    Returns:
        int: The n‑th prime.

    Raises:
        ValueError: If n is negative.

    Examples:
        >>> getPrime(0)
        2
        >>> getPrime(1)
        3
        >>> getPrime(3)
        7
    """
    if n < 0:
        raise ValueError("n must be ≥ 0")
    # sympy.prime is 1‑indexed: prime(1) = 2
    return sympy_prime(n + 1)


def getPrimesBetween(pNumber1: int, pNumber2: int) -> list[int]:
    """
    Return all primes strictly between two given primes.

    Args:
        pNumber1: Lower bound (must be prime).
        pNumber2: Upper bound (must be prime, > pNumber1).

    Returns:
        list[int]: All primes p with pNumber1 < p < pNumber2.

    Raises:
        ValueError: If inputs are not primes or pNumber1 ≥ pNumber2.

    Examples:
        >>> getPrimesBetween(3, 13)
        [5, 7, 11]
        >>> getPrimesBetween(2, 3)
        []
    """
    if not (isPrime(pNumber1) and isPrime(pNumber2)):
        raise ValueError("Both arguments must be prime numbers")
    if pNumber1 >= pNumber2:
        raise ValueError("pNumber1 must be less than pNumber2")
    return [p for p in range(pNumber1 + 1, pNumber2) if isPrime(p)]


# ---------- Prime factorization (accelerated by sympy) ----------


@lru_cache(maxsize=None)
def primeFactorization(number: int) -> list[int]:
    """
    Return the prime factors of `number` (with multiplicity) using SymPy's factorint.

    For number < 2, returns an empty list.

    Args:
        number: Non‑negative integer.

    Returns:
        list[int]: Prime factors in ascending order.

    Raises:
        ValueError: If number is negative.

    Examples:
        >>> primeFactorization(12)
        [2, 2, 3]
        >>> primeFactorization(1)
        []
        >>> primeFactorization(97)
        [97]
    """
    if number < 0:
        raise ValueError("number must be non‑negative")
    if number < 2:
        return []
    factors_dict = factorint(number)  # {prime: exponent}
    factors: list[int] = []
    for p, exp in sorted(factors_dict.items()):
        factors.extend([p] * exp)
    return factors  # returns a list – doctest expects list


def greatestPrimeFactor(number: int) -> int:
    """Return the largest prime factor of `number`."""
    if number < 2:
        raise ValueError("number must be ≥ 2")
    factors = primeFactorization(number)
    return max(factors)


def smallestPrimeFactor(number: int) -> int:
    """Return the smallest prime factor of `number`."""
    if number < 2:
        raise ValueError("number must be ≥ 2")
    factors = primeFactorization(number)
    return min(factors)


# ---------- GCD, LCM, Divisors, Perfect (using math.gcd) ----------


def gcd(number1: int, number2: int) -> int:
    """Greatest common divisor (uses math.gcd)."""
    if number1 < 0 or number2 < 0:
        raise ValueError("Arguments must be non‑negative")
    return math.gcd(number1, number2)


def kgV(number1: int, number2: int) -> int:
    """Least common multiple."""
    if number1 < 1 or number2 < 1:
        raise ValueError("Arguments must be positive")
    return abs(number1 * number2) // gcd(number1, number2)


def getDivisors(n: int) -> list[int]:
    """Return all positive divisors of `n` (including 1 and `n`)."""
    if n < 1:
        raise ValueError("n must be ≥ 1")
    return [d for d in range(1, n + 1) if n % d == 0]


def isPerfectNumber(number: int) -> bool:
    """Check if `number` is a perfect number."""
    if number < 2:
        raise ValueError("number must be ≥ 2")
    divisors = getDivisors(number)
    return sum(divisors[:-1]) == number


# ---------- Fraction simplification ----------


def simplifyFraction(numerator: int, denominator: int) -> tuple[int, int]:
    """Reduce a fraction to simplest form."""
    if denominator == 0:
        raise ValueError("denominator cannot be zero")
    g = gcd(abs(numerator), abs(denominator))
    return (numerator // g, denominator // g)


# ---------- Factorial and Fibonacci (optimized with math.factorial and caching) ----------


def factorial(n: int) -> int:
    """Compute n! using math.factorial (C implementation)."""
    if n < 0:
        raise ValueError("n must be ≥ 0")
    return math.factorial(n)


@lru_cache(maxsize=None)
def fib(n: int) -> int:
    """Return the n‑th Fibonacci number (0‑indexed: fib(0)=1, fib(1)=1)."""
    if n < 0:
        raise ValueError("n must be ≥ 0")
    if n < 2:
        return 1
    return fib(n - 1) + fib(n - 2)


# ---------- Goldbach's conjecture ----------


def goldbach(number: int) -> list[int]:
    """Find two primes summing to `number` (Goldbach's conjecture)."""
    if number <= 2 or not isEven(number):
        raise ValueError("number must be even and > 2")
    primes = sieveEr(number)
    prime_set = set(primes)
    for p in primes:
        q = number - p
        if q in prime_set:
            return [p, q]
    raise RuntimeError("Goldbach conjecture failed for this number")


# ---------- Pi calculation (unchanged, uses decimal) ----------


def pi(maxK: int = 70, prec: int = 1008, disp: int = 1007) -> str:
    """Compute π using the Chudnovsky algorithm (unchanged)."""
    from decimal import Decimal as Dec
    from decimal import getcontext as gc

    gc().prec = prec
    K, M, L, X, S = 6, 1, 13591409, 1, 13591409
    for k in range(1, maxK + 1):
        M = Dec((K**3 - (K << 4)) * M / k**3)
        L += 545140134
        X *= -262537412640768000
        S += Dec(M * L) / X
        K += 12
    pi_val = 426880 * Dec(10005).sqrt() / S
    return str(pi_val)[:disp]


# ---------- Run doctests ----------
if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
