"""Cartesian Product of Two Lists."""

# Import
from itertools import product


# Cartesian Product of Two Lists
def cartesian_product(list1, list2):
    """Cartesian Product of Two Lists."""
    for _i in list1:
        for _j in list2:
            print((_i, _j), end=' ')


# Main
if __name__ == '__main__':
    list1 = input().split()
    list2 = input().split()

    # Convert to ints
    list1 = [int(i) for i in list1]
    list2 = [int(i) for i in list2]

    cartesian_product(list1, list2)

