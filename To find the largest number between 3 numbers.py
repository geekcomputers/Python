"""Find the Largest Number Among Three Inputs

This program takes three numeric inputs from the user and determines
the largest number using a straightforward comparison approach.
"""


def get_valid_number(input_prompt: str) -> float:
    """Prompt the user for a number and validate the input.

    Args:
        input_prompt: String to display when requesting input

    Returns:
        Validated numeric value (integer or float)

    Raises:
        ValueError: If input cannot be converted to a number
    """
    while True:
        user_input = input(input_prompt).strip()
        try:
            return float(user_input)
        except ValueError:
            print(f"Error: '{user_input}' is not a valid number. Please try again.")


def get_three_numbers() -> list[float]:
    """Collect three valid numbers from the user.

    Returns:
        List containing three numeric values
    """
    numbers: list[float] = []
    for i in range(3):
        num = get_valid_number(f"Enter number {i + 1}: ")
        numbers.append(num)
    return numbers


def find_largest(numbers: list[int | float]) -> int | float:
    """Determine the largest number in a list of three numbers.

    Args:
        numbers: List containing exactly three numeric values

    Returns:
        The largest value in the list

    Raises:
        ValueError: If the input list does not contain exactly three numbers
        TypeError: If any element in the list is not numeric
    """
    # Validate input list
    if len(numbers) != 3:
        raise ValueError(f"Expected 3 numbers, got {len(numbers)}")

    for i, num in enumerate(numbers):
        if not isinstance(num, (int, float)):
            raise TypeError(f"Element {i + 1} is not a number: {type(num).__name__}")

    return max(numbers)


def main() -> None:
    """Main function to coordinate input collection and result display."""
    print("Find the largest number among three inputs\n")

    try:
        numbers = get_three_numbers()
        largest = find_largest(numbers)
        print(f"\nThe largest among the three numbers is: {largest}")
    except (ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import sys  # Import here to avoid unused import in module mode

    main()
