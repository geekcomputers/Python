def main():
    """Main function that gets user input and calls the pattern generation function."""
    try:
        lines = int(input("Enter number of lines: "))
        if lines < 0:
            raise ValueError("Number of lines must be non-negative")
        result = pattern(lines)
        print(result)
    except ValueError as e:
        if "invalid literal" in str(e):
            print(f"Error: Please enter a valid integer number. {e}")
        else:
            print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def pattern(lines: int) -> str:
    """
    Generate a pattern of '@' and '$' characters.

    Args:
        lines: Number of lines to generate in the pattern

    Returns:
        A multiline string containing the specified pattern

    Raises:
        ValueError: If lines is negative

    Examples:
    >>> print(pattern(3))
    @@@    $
    @@    $$
    @    $$$

    >>> print(pattern(1))
    @    $

    >>> print(pattern(0))
    <BLANKLINE>

    >>> pattern(-5)
    Traceback (most recent call last):
    ...
    ValueError: Number of lines must be non-negative
    """
    if lines < 0:
        raise ValueError("Number of lines must be non-negative")
    if lines == 0:
        return ""

    pattern_lines = []
    for i in range(lines, 0, -1):
        at_pattern = "@" * i
        dollar_pattern = "$" * (lines - i + 1)
        pattern_lines.append(f"{at_pattern}    {dollar_pattern}")

    return "\n".join(pattern_lines)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
    main()
