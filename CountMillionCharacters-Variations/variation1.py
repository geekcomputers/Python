"""Character Count Utility

A program that counts the frequency of each uppercase character in a specified file.
Handles user input gracefully and provides clear error messages for missing files.
Compatible with Python 3.13.5 and all modern Python 3 versions.
"""



def count_chars(filename: str) -> dict[str, int]:
    """Count the frequency of each uppercase character in a file.

    Args:
        filename: Path to the file to be analyzed.

    Returns:
        A dictionary where keys are uppercase characters and values are their counts.
        Includes all whitespace, punctuation, and special characters present in the file.
    """
    char_counts: dict[str, int] = {}

    with open(filename) as file:  # Open file in read mode
        content: str = file.read()
        for char in content.upper():  # Convert to uppercase to ensure case insensitivity
            # Update count for current character (default to 0 if not found)
            char_counts[char] = char_counts.get(char, 0) + 1

    return char_counts


def main() -> None:
    """Main function to handle user interaction and coordinate the character counting process.

    Prompts the user for a filename, processes the file, and displays character counts.
    Allows the user to exit by entering '0'. Handles missing files with friendly error messages.
    """
    print("Character Count Utility")
    print("Enter filename to analyze (or '0' to exit)\n")

    while True:
        try:
            # Get user input with prompt
            user_input: str = input("File name / (0)exit: ").strip()
            
            # Check for exit condition
            if user_input == "0":
                print("Exiting program. Goodbye!")
                break
            
            # Process file and display results
            counts: dict[str, int] = count_chars(user_input)
            print(f"Character counts for '{user_input}':")
            print(counts)
            print()  # Add blank line for readability

        except FileNotFoundError:
            print(f"Error: File '{user_input}' not found. Please try again.\n")
        except OSError as e:
            print(f"Error reading file: {str(e)}. Please try again.\n")


if __name__ == "__main__":
    main()