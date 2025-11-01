"""
Leap Year Checker.

Determine whether a given year is a leap year.

Doctests:

>>> is_leap_year(2000)
True
>>> is_leap_year(1900)
False
>>> is_leap_year(2024)
True
>>> is_leap_year(2023)
False
"""


def is_leap_year(year: int) -> bool:
    """
    Return True if year is a leap year, False otherwise.

    Rules:
    - Divisible by 4 => leap year
    - Divisible by 100 => not leap year
    - Divisible by 400 => leap year
    """
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    year_input = input("Enter a year: ").strip()
    try:
        year = int(year_input)
        if is_leap_year(year):
            print(f"{year} is a leap year")
        else:
            print(f"{year} is not a leap year")
    except ValueError:
        print("Invalid input! Please enter a valid integer year.")
