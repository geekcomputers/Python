import os
import sys
import time
from typing import NoReturn


def _get_invalid_filename_chars() -> tuple[str, ...]:
    """Return a tuple of invalid filename characters for the current OS."""
    if sys.platform.startswith('win'):
        return ('\\', '/', ':', '*', '?', '"', '<', '>', '|')
    else:  # Linux/macOS
        return ('/',)


def is_valid_filename(filename: str) -> tuple[bool, str]:
    """
    Validate if a filename is valid for the current operating system.
    
    Args:
        filename: Filename to validate
        
    Returns:
        Tuple containing (is_valid: bool, error_message: str)
    """
    if not filename.strip():
        return False, "Filename cannot be empty or contain only whitespace"
    
    invalid_chars = _get_invalid_filename_chars()
    for char in filename:
        if char in invalid_chars:
            return False, f"Contains invalid character: '{char}' (invalid: {', '.join(invalid_chars)})"
    
    max_length = 255 if not sys.platform.startswith('win') else 260
    if len(filename) > max_length:
        return False, f"Too long (max {max_length} characters)"
    
    if sys.platform.startswith('win'):
        reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 
                         'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 
                         'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
        base_name = filename.split('.')[0].upper()
        if base_name in reserved_names:
            return False, f"'{filename}' is a reserved system name"
    
    return True, "Valid filename"


def write_to_file(file_name: str) -> None:
    """Create a new file and allow user to add content line by line."""
    if os.path.exists(file_name):
        print(f"Error: File '{file_name}' already exists (preventing overwrite).")
        return

    parent_dir = os.path.dirname(file_name)
    if parent_dir and not os.path.exists(parent_dir):
        print(f"Error: Parent directory '{parent_dir}' does not exist.")
        return

    try:
        with open(file_name, "a", encoding="utf-8") as file:
            print(f"Created file: '{file_name}'")
            
            while True:
                text = input("Enter text to add (press Enter to skip): ").rstrip('\n')
                
                if text:
                    file.write(f"{text}\n")
                    print(f"Added: {text}")
                else:
                    print("Note: Empty input will not be written.")

                while True:
                    choice = input("Add more content? (y/n): ").strip().lower()
                    if choice in ('y', 'n'):
                        break
                    print("Invalid input. Please enter 'y' or 'n'.")
                
                if choice == 'n':
                    print("Content writing completed.")
                    break

    except PermissionError:
        print(f"Error: Permission denied for '{file_name}'.")
    except OSError as e:
        print(f"System error while writing: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def print_short_lines(file_name: str) -> None:
    """Read a file and print lines with length < 50 characters (excluding newline)."""
    if not os.path.exists(file_name):
        print(f"Error: File '{file_name}' does not exist.")
        return

    if not os.path.isfile(file_name):
        print(f"Error: '{file_name}' is a directory, not a file.")
        return

    try:
        with open(file_name, encoding="utf-8") as file:
            lines: list[str] = file.readlines()
            if not lines:
                print(f"Info: File '{file_name}' is empty.")
                return

            short_lines: list[str] = [line for line in lines if len(line.rstrip('\n')) < 50]
            
            if not short_lines:
                print("No lines with length < 50 characters (excluding newline).")
            else:
                print(f"\nLines in '{file_name}' with length < 50:")
                for line_num, line in enumerate(short_lines, 1):
                    print(f"Line {line_num}: {line.rstrip('\n')}")

    except PermissionError:
        print(f"Error: Permission denied to read '{file_name}'.")
    except UnicodeDecodeError:
        print(f"Error: '{file_name}' is not a valid text file.")
    except OSError as e:
        print(f"System error while reading: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def get_valid_filename() -> str:
    """Prompt user for a filename until a valid one is provided."""
    while True:
        file_name = input("Enter the name of the file to create: ").strip()
        print(f"Selected filename: '{file_name}'")

        is_valid, msg = is_valid_filename(file_name)
        if is_valid:
            return file_name
        print(f"Invalid filename: {msg}. Please try again.\n")


def main() -> NoReturn:
    """Main function coordinating file operations with retry for invalid filenames."""
    # Get valid filename with retry mechanism
    file_name = get_valid_filename()
    
    # Write content to file
    write_to_file(file_name)
    
    # Brief pause to ensure file operations complete
    time.sleep(1)
    
    # Read and display short lines
    print_short_lines(file_name)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
