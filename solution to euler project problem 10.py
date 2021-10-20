##author-slayking1965
#"""
#https://projecteuler.net/problem=10
#Problem Statement:
#The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
#Find the sum of all the primes below two million using Sieve_of_Eratosthenes:
#The sieve of Eratosthenes is one of the most efficient ways to find all primes
#smaller than n when n is smaller than 10 million.  Only for positive numbers.
#Find the sum of all the primes below two million.
#"""


#def prime_sum(n: int) -> int:
#    """Returns the sum of all the primes below n.
#def solution(n: int = 2000000) -> int:
#    """Returns the sum of all the primes below n using Sieve of Eratosthenes:
#    https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
#    The sieve of Eratosthenes is one of the most efficient ways to find all primes
#    smaller than n when n is smaller than 10 million.  Only for positive numbers.
#    >>> prime_sum(2_000_000)
#    >>> solution(2_000_000)
#    142913828922
#    >>> prime_sum(1_000)
#    >>> solution(1_000)
#    76127
#    >>> prime_sum(5_000)
#    >>> solution(5_000)
#    1548136
#    >>> prime_sum(10_000)
#    >>> solution(10_000)
#    5736396
#    >>> prime_sum(7)
#    >>> solution(7)
#    10
#    >>> prime_sum(7.1)  # doctest: +ELLIPSIS
#    >>> solution(7.1)  # doctest: +ELLIPSIS
#    Traceback (most recent call last):
#    ...
#    TypeError: 'float' object cannot be interpreted as an integer
#    >>> prime_sum(-7)  # doctest: +ELLIPSIS
#    >>> solution(-7)  # doctest: +ELLIPSIS
#    Traceback (most recent call last):
#    ...
#    IndexError: list assignment index out of range
#    >>> prime_sum("seven")  # doctest: +ELLIPSIS
#    >>> solution("seven")  # doctest: +ELLIPSIS
#    Traceback (most recent call last):
#    ...
#    TypeError: can only concatenate str (not "int") to str
#    """
#    list_ = [0 for i in range(n + 1)]
#    list_[0] = 1
#    list_[1] = 1
#    primality_list = [0 for i in range(n + 1)]
#    primality_list[0] = 1
#    primality_list[1] = 1

#    for i in range(2, int(n ** 0.5) + 1):
#        if list_[i] == 0:
#        if primality_list[i] == 0:
#            for j in range(i * i, n + 1, i):
#                list_[j] = 1
#    s = 0
#                primality_list[j] = 1
#    sum_of_primes = 0
#    for i in range(n):
#        if list_[i] == 0:
#            s += i
#    return s
#        if primality_list[i] == 0:
#            sum_of_primes += i
#    return sum_of_primes


#if __name__ == "__main__":
#    # import doctest
#    # doctest.testmod()
#    print(prime_sum(int(input().strip())))
#    print(solution(int(input().strip())))
