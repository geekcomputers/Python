#!/usr/bin/env python3
"""
Luhn Algorithm – Calculate the check digit for a given payload.

This module provides a function to compute the Luhn check digit
for a numeric string and a command-line interface to demonstrate it.
"""

import sys


def luhn_checksum(payload: str) -> int:
    """
    Compute the Luhn check digit for the given payload string.

    The algorithm processes digits from right to left, doubling every
    second digit starting from the rightmost digit (i.e., positions
    1, 3, 5, ... from the right). If doubling results in a number
    greater than 9, subtract 9 from it (equivalent to summing the digits).

    Args:
        payload: A string of decimal digits (e.g., "7992739871").

    Returns:
        The check digit (0–9) that makes the full number valid.

    Raises:
        ValueError: If payload contains any non-digit character.
    """
    if not payload.isdigit():
        raise ValueError("Payload must contain only digits.")

    digits = [int(ch) for ch in payload]
    total = 0

    # Iterate from the rightmost digit, starting index at 1
    for i, d in enumerate(reversed(digits), start=1):
        if i % 2 == 1:  # Odd position from the right (1st, 3rd, 5th, ...)
            doubled = d * 2
            total += doubled if doubled < 10 else doubled - 9
        else:
            total += d

    # The check digit is the number that makes the total a multiple of 10
    check_digit = (10 - total % 10) % 10
    return check_digit


def main() -> None:
    """Command-line entry point: prompts for a 10-digit payload and prints the result."""
    try:
        payload = input("Enter number to validate (e.g., 7992739871): ").strip()
        if len(payload) != 10:
            print("Error: Number must be exactly 10 digits.")
            sys.exit(1)

        check = luhn_checksum(payload)
        full_number = payload + str(check)

        print(f"Sum of all digits: {sum(int(ch) for ch in payload)}")
        print(f"Check digit: {check}")
        print(f"Full valid number (11 digits): {full_number}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    input("Press Enter to exit...")


# =========================== Pytest Tests ===========================


def test_luhn_checksum_known_case() -> None:
    """Known example from Wikipedia: 7992739871 → check digit 3."""
    assert luhn_checksum("7992739871") == 3


def test_luhn_checksum_zero_checkdigit() -> None:
    """Case where the check digit is 0: using 2222222222."""
    # For 2222222222, total = 30 → (10 - 30%10)%10 = 0
    assert luhn_checksum("2222222222") == 0


def test_luhn_checksum_another_payload() -> None:
    """Another arbitrary payload: 1234567890 → check digit 3."""
    assert luhn_checksum("1234567890") == 3


def test_luhn_checksum_invalid_characters() -> None:
    """Non-digit input must raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Payload must contain only digits."):
        luhn_checksum("1234abc567")


def test_luhn_checksum_empty_string() -> None:
    """Empty string must raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Payload must contain only digits."):
        luhn_checksum("")


# =========================== Entry Point ===========================

if __name__ == "__main__":
    # If the --test argument is given, run pytest on this file.
    if "--test" in sys.argv:
        import pytest

        sys.exit(pytest.main([__file__]))
    else:
        main()
