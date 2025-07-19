from collections import Counter
from pathlib import Path


def count_lowercase_chars(file_path: str | Path = "happy.txt") -> tuple[int, int, int]:
    """
    Counts the number of lowercase letters, uppercase letters, and total alphabetic characters in a text file.

    Args:
        file_path (str | Path): Path to the text file. Defaults to "happy.txt".

    Returns:
        tuple[int, int, int]: A tuple containing the count of lowercase letters, uppercase letters, and total alphabetic characters.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the file cannot be accessed.
    """
    try:
        with open(file_path, encoding="utf-8") as file:
            content = file.read()
            Counter(content)

            lowercase_count = sum(1 for char in content if char.islower())
            uppercase_count = sum(1 for char in content if char.isupper())
            total_letters = lowercase_count + uppercase_count

            return lowercase_count, uppercase_count, total_letters
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.")
    except PermissionError:
        raise PermissionError(f"Permission denied to read '{file_path}'.")


def main() -> None:
    """Main function to execute the character counting and display results."""
    try:
        file_path = Path("happy.txt")
        lowercase, uppercase, total = count_lowercase_chars(file_path)

        print(f"The total number of lowercase letters is: {lowercase}")
        print(f"The total number of uppercase letters is: {uppercase}")
        print(f"The total number of alphabetic characters is: {total}")

    except (FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
