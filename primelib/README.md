
# Prime Number Utilities

**Free open-source library** – originally by Christian Bender, now accelerated with **SymPy** and fully type‑hinted.

This Python library provides a comprehensive set of functions for dealing with prime numbers and general number theory:
- Primality testing
- Prime generation (Sieve of Eratosthenes, nth prime, interval primes)
- Prime factorization (with multiplicity)
- Greatest / smallest prime factor
- GCD & LCM
- Divisor enumeration
- Perfect number detection
- Fraction simplification
- Factorial and Fibonacci
- Goldbach's conjecture
- High‑precision π calculation (Chudnovsky algorithm)

---

## 📦 Requirements

- Python 3.8 or newer (for type hints)
- [SymPy](https://www.sympy.org/) – for fast number‑theoretic operations

Install SymPy with:

```bash
pip install sympy
```

On Termux (Android), you can also try:

```bash
pkg install python-sympy
```

or simply use pip.

---

🚀 Installation

Just copy primelib.py into your project directory, then import it:

```python
import primelib
# or
from primelib import *
```

---

📖 Function Reference

Function Description
isPrime(number) Returns True if number is prime, otherwise False.
sieveEr(N) Returns a list of all primes ≤ N (Sieve of Eratosthenes).
getPrimeNumbers(N) Alias for sieveEr.
getPrime(n) Returns the n‑th prime (0‑based: 0 → 2, 1 → 3, 3 → 7, …).
getPrimesBetween(p1, p2) Returns all primes strictly between primes p1 and p2.
primeFactorization(number) Returns a list of prime factors with multiplicity (e.g., 12 → [2, 2, 3]).
greatestPrimeFactor(number) Largest prime factor.
smallestPrimeFactor(number) Smallest prime factor.
gcd(a, b) Greatest common divisor (non‑negative integers).
kgV(a, b) Least common multiple (positive integers).
getDivisors(n) All positive divisors of n (including 1 and n).
isPerfectNumber(number) True if number equals the sum of its proper divisors.
simplifyFraction(num, den) Reduces a fraction to lowest terms, returns (num, den).
factorial(n) n! (uses C‑level math.factorial).
fib(n) n‑th Fibonacci number (0‑based: fib(0)=1, fib(1)=1).
goldbach(number) Returns two primes summing to even number > 2.
pi(maxK=70, prec=1008, disp=1007) Computes π using Chudnovsky algorithm, returns a string.

---

💻 Usage Examples

```python
import primelib

# Primality test
print(primelib.isPrime(13))          # True
print(primelib.isPrime(100))         # False

# Prime factorization
print(primelib.primeFactorization(40))   # [2, 2, 2, 5]

# Get the 5th prime (0‑based → index 5 → 13)
print(primelib.getPrime(5))          # 13

# All primes between 10 and 30
print(primelib.getPrimesBetween(10, 30))  # [11, 13, 17, 19, 23, 29]

# GCD and LCM
print(primelib.gcd(48, 18))          # 6
print(primelib.kgV(48, 18))          # 144

# Divisors and perfect number
print(primelib.getDivisors(28))      # [1, 2, 4, 7, 14, 28]
print(primelib.isPerfectNumber(28))  # True

# Fraction simplification
print(primelib.simplifyFraction(12, 8))  # (3, 2)

# Factorial and Fibonacci
print(primelib.factorial(5))         # 120
print(primelib.fib(5))               # 8

# Goldbach
print(primelib.goldbach(28))         # [5, 23]

# Pi (first 10 digits)
print(primelib.pi(5, 20, 12))        # 3.1415926535
```

---

🧪 Testing

The module includes doctests embedded in each function’s docstring.
Run them with:

```bash
python -m doctest -v primelib.py
```

Or simply execute:

```bash
python primelib.py
```

to run all tests verbosely.

---

📝 API Documentation (Detailed)

You can view the full docstring of any function with Python’s help():

```python
help(primelib.isPrime)
help(primelib.primeFactorization)
```