def fib(n):
    return n if n in [0, 1] else fib(n - 1) + fib(n - 2)
