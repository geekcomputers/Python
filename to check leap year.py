"""Leap Year Checker

This program determines if a given year is a leap year based on the
Gregorian calendar rules:
1. The year must be evenly divisible by 4
2. If the year can also be evenly divided by 100, it is not a leap year
   unless...
3. The year is also evenly divisible by 400, then it is a leap year
"""


def is_leap_year(year: int) -> bool:
    """Determine if a given year is a leap year.

    Args:
        year: Integer representing the year to check

    Returns:
        True if the year is a leap year, False otherwise

    Examples:
        >>> is_leap_year(2000)
        True
        >>> is_leap_year(1900)
        False
        >>> is_leap_year(2024)
        True
    """
    if not isinstance(year, int):
        raise TypeError("Year must be an integer")

    if year <= 0:
        raise ValueError("Year must be a positive integer")

    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def main() -> None:
    """Main function to handle user input and display results"""
    try:
        # Get user input
        year_input = input("Enter a year (e.g., 2024): ").strip()

        # Validate and convert input to integer
        try:
            year = int(year_input)
        except ValueError:
            raise ValueError(f"Invalid year: '{year_input}'. Please enter an integer.")

        # Check and display result
        result = is_leap_year(year)
        print(f"{year} is a leap year" if result else f"{year} is not a leap year")

    except (TypeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import sys

    # Import here to avoid unused import in module mode
    main()
