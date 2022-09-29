def fib(term):
    return term if term <= 1 else fib(term - 1) + fib(term - 2)


# Change this value to adjust the number of terms in the sequence.
number_of_terms = int(input())
for i in range(number_of_terms):
    print(fib(i))
