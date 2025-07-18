from pathlib import Path
from typing import Tuple

def count_lowercase_chars(file_path: str | Path) -> Tuple[int, int, int]:
    """
    Counts lowercase, uppercase, and total alphabetic characters in a text file.

    Args:
        file_path: Path to the text file (can be a string or Path object).

    Returns:
        Tuple containing (lowercase count, uppercase count, total alphabetic count).

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        PermissionError: If access to the file is denied.
        IsADirectoryError: If the path points to a directory instead of a file.
    """
    # Convert to Path object for consistent path handling
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    # Check if it's a file (not a directory)
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")

    # Read file and count characters
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        lowercase = sum(1 for c in content if c.islower())
        uppercase = sum(1 for c in content if c.isupper())
        total_alpha = lowercase + uppercase
    
    return lowercase, uppercase, total_alpha

def main() -> None:
    """Main function to execute the file check and character counting."""
    # Specify the path to happy.txt (update this to your actual path!)
    # Example for Windows: r"C:\Users\YourName\1 File handle\File handle text\happy.txt"
    # Example for macOS/Linux: "/home/YourName/1 File handle/File handle text/happy.txt"
    file_path = r"1 File handle\File handle text\happy.txt"  # Update this path!

    try:
        # Print current working directory for debugging
        print(f"Current working directory: {Path.cwd()}")
        
        lowercase, uppercase, total_alpha = count_lowercase_chars(file_path)
        
        print("\n=== Character Count Results ===")
        print(f"Lowercase letters: {lowercase}")
        print(f"Uppercase letters: {uppercase}")
        print(f"Total alphabetic characters: {total_alpha}")
    
    except FileNotFoundError as e:
        print(f"\nError: {e}. Please check the file path.")
    except IsADirectoryError as e:
        print(f"\nError: {e}")
    except PermissionError:
        print(f"\nError: No permission to access {file_path}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()