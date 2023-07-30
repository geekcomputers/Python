def fibonacci_generator(n=None):
    """
    Generating function up to n fibonacci numbers iteratively
    Params:
        n: int
    Return:
        int
    """


def fibonacci_generator(n=None):
    """
    Generating function for up to n Fibonacci numbers iteratively.
    Params:
        n: int
    Return:
        int
    """
    f0, f1 = 0, 1
    yield f1
    while n is None or n > 1:
        fn = f0 + f1
        yield fn
        f0, f1 = f1, fn
        if n is not None:
            n -= 1


if __name__ == "__main__":
    for n_fibo in fibonacci_generator(7):
        print(n_fibo)
