#!/usr/bin/env python3
"""
Luhn Algorithm – Credit Card Number Validation

This module provides a function to validate a card number using the Luhn algorithm.
It also includes a command-line interface and a set of unit tests.
"""

import sys


def clean_card_number(card_number: str) -> str:
    """
    Remove all non-digit characters from the input string.

    Args:
        card_number: Raw input possibly containing spaces, hyphens, etc.

    Returns:
        A string containing only digits.

    Raises:
        ValueError: If after cleaning no digits remain.
    """
    cleaned = "".join(ch for ch in card_number if ch.isdigit())
    if not cleaned:
        raise ValueError("Card number must contain at least one digit.")
    return cleaned


def validate_luhn(card_number: str) -> bool:
    """
    Verify whether a given card number (as a string of digits) passes the Luhn check.

    The algorithm:
        - Reverse the digits.
        - Sum the digits in odd positions (1st, 3rd, ... from the right) as-is.
        - For digits in even positions (2nd, 4th, ... from the right), double them;
          if the result is >9, subtract 9.
        - The card is valid if the total sum is a multiple of 10.

    Args:
        card_number: A string of digits (no separators).

    Returns:
        True if the number passes the Luhn check, False otherwise.

    Raises:
        ValueError: If card_number contains non-digit characters.
    """
    if not card_number.isdigit():
        raise ValueError("Card number must contain only digits.")

    # Work from the rightmost digit
    reversed_digits = [int(d) for d in card_number[::-1]]

    # Sum digits at odd positions (1-indexed from the right)
    odd_sum = sum(reversed_digits[0::2])  # indices 0,2,4,...

    # Sum digits at even positions after doubling and subtracting 9 if >=10
    even_sum = 0
    for d in reversed_digits[1::2]:  # indices 1,3,5,...
        doubled = d * 2
        even_sum += doubled if doubled < 10 else doubled - 9

    total = odd_sum + even_sum
    return total % 10 == 0


def main() -> None:
    """Command-line entry point. Reads a card number from the user and prints validation result."""
    try:
        raw = input("Enter a credit card number (e.g., 4111-1111-4555-1142): ").strip()
        cleaned = clean_card_number(raw)

        if validate_luhn(cleaned):
            print("VALID!")
        else:
            print("INVALID!")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =========================== Tests (pytest) ===========================


# =========================== Tests (pytest) ===========================


def test_clean_card_number():
    """Test cleaning function."""
    import pytest

    assert clean_card_number("4111-1111-4555-1142") == "4111111145551142"
    assert clean_card_number("  123  ") == "123"
    with pytest.raises(
        ValueError, match="Card number must contain at least one digit."
    ):
        clean_card_number("")


def test_validate_luhn_valid():
    """Known valid cards."""
    assert validate_luhn("4111111111111111") is True
    assert validate_luhn("5555555555554444") is True
    assert validate_luhn("378282246310005") is True


def test_validate_luhn_invalid():
    """Known invalid card numbers."""
    assert validate_luhn("1234567890") is False
    assert validate_luhn("4111111111111112") is False


def test_validate_luhn_non_digit():
    """Should raise ValueError on non-digit input."""
    import pytest

    with pytest.raises(ValueError, match="only digits"):
        validate_luhn("1234abc")


if __name__ == "__main__":
    if "--test" in sys.argv:
        import pytest

        sys.exit(pytest.main([__file__]))
    else:
        main()
