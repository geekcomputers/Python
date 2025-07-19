import os
import sys
from typing import NoReturn


def count_lowercase_letters(file_path: str) -> tuple[int, int]:
    """
    Count lowercase and uppercase alphabetic characters in a text file.
    
    Reads the entire content of the file, iterates through each character,
    and counts letters that are lowercase (a-z) and uppercase (A-Z). Non-alphabetic
    characters (numbers, symbols, whitespace) are ignored.
    
    Args:
        file_path: Path to the text file (e.g., "happy.txt")
        
    Returns:
        Tuple containing:
            - Count of lowercase letters (a-z)
            - Count of uppercase letters (A-Z)
            
    Raises:
        FileNotFoundError: If the specified file does not exist
        PermissionError: If the user lacks permission to read the file
        IsADirectoryError: If the path points to a directory instead of a file
        UnicodeDecodeError: If the file contains non-UTF-8 encoded data
        OSError: For other system-related errors (e.g., invalid path)
    """
    lowercase_count: int = 0
    uppercase_count: int = 0

    # Read file content with explicit encoding
    with open(file_path, encoding='utf-8') as file:
        content: str = file.read()

    # Iterate through each character to count letters
    for char in content:
        if char.islower():
            lowercase_count += 1
        elif char.isupper():
            uppercase_count += 1

    return lowercase_count, uppercase_count


def validate_file_path(file_path: str) -> bool:
    """
    Validate if the provided file path points to an existing, readable file.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if the path is valid and points to a readable file; False otherwise
    """
    # Check if path is empty
    if not file_path.strip():
        print("Error: File path cannot be empty.")
        return False

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False

    # Check if path is a file (not directory)
    if not os.path.isfile(file_path):
        print(f"Error: '{file_path}' is a directory, not a file.")
        return False

    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        print(f"Error: No permission to read '{file_path}'.")
        return False

    return True


def main() -> NoReturn:
    """
    Main function to handle user interaction and coordinate the counting process.
    
    Guides the user to input a file path, validates it, triggers the counting function,
    and displays the results with clear formatting.
    """
    print("=== Alphabet Case Counter ===")
    print("This program counts lowercase (a-z) and uppercase (A-Z) letters in a text file.\n")

    # Get and validate file path from user
    while True:
        file_path: str = input("Enter the path to the text file (e.g., 'happy.txt'): ").strip()
        if validate_file_path(file_path):
            break
        print("Please try again.\n")

    try:
        # Count lowercase and uppercase letters
        lowercase_count, uppercase_count = count_lowercase_letters(file_path)

        # Display results
        print(f"\nStatistics for '{os.path.basename(file_path)}':")
        print(f"Lowercase letters (a-z): {lowercase_count}")
        print(f"Uppercase letters (A-Z): {uppercase_count}")
        print(f"Total alphabetic letters: {lowercase_count + uppercase_count}")

    except UnicodeDecodeError:
        print(f"\nError: '{file_path}' contains non-UTF-8 data. Cannot read as text file.")
    except OSError as e:
        print(f"\nSystem error: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        print("\nOperation completed.")
        sys.exit(0)


if __name__ == "__main__":
    main()