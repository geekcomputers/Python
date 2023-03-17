def lcm(x, y):
    """
        Find least common multiple of 2 positive integers.
        :param x: int - first integer
        :param y: int - second integer
        :return: int - least common multiple

        >>> lcm(8, 4)
            8
        >>> lcm(5, 3)
            15
        >>> lcm(15, 9)
            45
        >>> lcm(124, 23)
            2852
        >>> lcm(3, 6)
            6
        >>> lcm(13, 34)
            442
        >>> lcm(235, 745)
            35015
        >>> lcm(65, 86)
            5590
        >>> lcm(0, 1)
            -1
        >>> lcm(-12, 35)
            -1
    """
    if x <= 0 or y <= 0:
        return -1

    if x > y:
        greater_number = x
    else:
        greater_number = y

    while True:
        if (greater_number % x == 0) and (greater_number % y == 0):
            lcm = greater_number
            break
        greater_number += 1
    return lcm


num_1 = int(input("Enter first number: "))
num_2 = int(input("Enter second number: "))

print(
    "The L.C.M. of "
    + str(num_1)
    + " and "
    + str(num_2)
    + " is "
    + str(lcm(num_1, num_2))
)
