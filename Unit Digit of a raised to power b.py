def last_digit(a, b):
    if b == 0:  # This Code assumes that 0^0 is 1
        return 1
    elif a % 10 in [0, 5, 6, 1]:
        return a % 10
    elif b % 4 == 0:
        return ((a % 10) ** 4) % 10
    else:
        return ((a % 10) ** (b % 4)) % 10


# Courtesy to https://brilliant.org/wiki/finding-the-last-digit-of-a-power/
