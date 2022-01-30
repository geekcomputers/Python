max_size = 10

print(
    "(a)"
    + " " * (max_size)
    + "(b)"
    + " " * (max_size)
    + "(c)"
    + " " * (max_size)
    + "(d)"
    + " " * (max_size)
)

for i in range(1, max_size + 1):

    print("*" * i, end=" " * (max_size - i + 3))

    print("*" * (max_size - i + 1), end=" " * (i - 1 + 3))

    print(" " * (i - 1) + "*" * (max_size - i + 1), end=" " * 3)

    print(" " * (max_size - i) + "*" * i)
