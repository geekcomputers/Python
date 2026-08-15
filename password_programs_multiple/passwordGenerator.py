#!/usr/bin/env python3
"""
Secure Password Generator – copies password to clipboard, never prints it.

This module generates a strong, memorable password by combining:
- Two random words (animal + colour)
- A random 3‑digit number
- A random special character

It also picks a random country and language (via pycountry) to provide a
memoization hint – these are NOT part of the password.

Security:
- Uses `secrets` module for cryptographically strong randomness.
- Does NOT print the password to the terminal (only a confirmation message).
- Copies the password to the system clipboard using `pyperclip`.

Author: Modified from GGearing / Prince Gangurde
Date: 2026-07-11
"""

import secrets
import string
from typing import Optional

# Optional imports with fallback
try:
    import pyperclip  # type: ignore
except ImportError:
    pyperclip = None

try:
    import pycountry
except ImportError:
    pycountry = None


# ----------------------------------------------------------------------
# Word lists – extend as needed, but keep them diverse
# ----------------------------------------------------------------------
ANIMALS = (
    "ant",
    "bear",
    "cat",
    "dog",
    "eagle",
    "fox",
    "goat",
    "hawk",
    "ibis",
    "jaguar",
    "kangaroo",
    "lion",
    "monkey",
    "newt",
    "owl",
    "panda",
    "quail",
    "rabbit",
    "shark",
    "tiger",
    "unicorn",
    "vulture",
    "wolf",
    "xerus",
    "yak",
    "zebra",
)

COLOURS = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "indigo",
    "violet",
    "purple",
    "magenta",
    "cyan",
    "pink",
    "brown",
    "white",
    "grey",
    "black",
)

SPECIAL_CHARS = "!@#$%/?<>|&*-=+_"
DIGITS = string.digits


# ----------------------------------------------------------------------
# Core function
# ----------------------------------------------------------------------
def generate_secure_password(
    animal_list: tuple = ANIMALS,
    colour_list: tuple = COLOURS,
    special_chars: str = SPECIAL_CHARS,
    num_digits: int = 3,
) -> str:
    """
    Generate a secure, memorable password using cryptographically strong randomness.

    The password is built as: <colour><digits><animal><special>
    One of the words (colour or animal) is randomly capitalised.

    Args:
        animal_list: Tuple of animal names.
        colour_list: Tuple of colour names.
        special_chars: String of allowed special characters.
        num_digits: Number of digits to include (default 3).

    Returns:
        The generated password string (not printed to console).

    Raises:
        ValueError: If any word list is empty.
    """
    if not animal_list or not colour_list or not special_chars:
        raise ValueError("Word lists and special chars must not be empty.")

    # Select random elements using secrets (cryptographically secure)
    animal = secrets.choice(animal_list)
    colour = secrets.choice(colour_list)

    # Build a random digit string of given length
    digit_str = "".join(secrets.choice(DIGITS) for _ in range(num_digits))

    special = secrets.choice(special_chars)

    # Randomly choose which word to uppercase
    if secrets.choice([True, False]):
        colour = colour.upper()
    else:
        animal = animal.upper()

    # Assemble the password
    password = f"{colour}{digit_str}{animal}{special}"
    return password


def get_random_country_and_language() -> tuple[Optional[str], Optional[str]]:
    """
    Return a random country name and a random language name (for memorisation hints).

    Falls back gracefully if pycountry is not installed.

    Returns:
        A tuple (country_name, language_name) – either may be None.
    """
    country = None
    language = None

    if pycountry is not None:
        try:
            # Pick a random country
            countries = list(pycountry.countries)
            if countries:
                country = secrets.choice(countries).name

            # Pick a random language (only those with a 'name' attribute)
            languages = [
                lang.name for lang in pycountry.languages if hasattr(lang, "name")
            ]
            if languages:
                language = secrets.choice(languages)
        except Exception:
            # Silently ignore any pycountry errors
            pass

    return country, language


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to the system clipboard using pyperclip.

    Args:
        text: The string to copy.

    Returns:
        True if successful, False if pyperclip is not available or fails.
    """
    if pyperclip is None:
        return False
    try:
        pyperclip.copy(text)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    """
    Generate a password, copy it to clipboard, and show hints.

    The password itself is never printed – only a confirmation message.
    """
    print("🔐 Generating a secure password...")

    # Generate the pass
