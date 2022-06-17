def is_square_free(factors):
    """
    This functions takes a list of prime factors as input.
    returns True if the factors are square free.
    """
    return all(factors.count(i) <= 1 for i in factors)


def prime_factors(n):
    """
    Returns prime factors of n as a list.
    """
    i = 2
    factors = []
    while i**2 <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def mobius_function(n):
    """
    Defines Mobius function
    """
    factors = prime_factors(n)
    if is_square_free(factors):
        return 1 if len(factors) % 2 == 0 else -1
    else:
        return 0
