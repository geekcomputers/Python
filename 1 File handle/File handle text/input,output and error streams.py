import sys
from typing import NoReturn


def read_and_print_file(file_path: str) -> None:
    """
    Read a text file and print its contents line by line.
    
    Reads the entire file content, prints each line to standard output,
    and sends an "End of file reached" message to standard error upon completion.
    
    Args:
        file_path: Path to the text file to be read.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the user lacks permission to read the file.
        IsADirectoryError: If the provided path points to a directory instead of a file.
        UnicodeDecodeError: If the file contains non-UTF-8 encoded data.
        OSError: For other OS-related errors (e.g., disk full).
    """
    try:
        # Open file in read mode with explicit UTF-8 encoding
        with open(file_path, encoding='utf-8') as file:
            # Read and print lines in chunks for better memory efficiency
            for line in file:
                sys.stdout.write(line)
                
            # Notify end of file via standard error stream
            sys.stderr.write("End of file reached\n")
            
    except FileNotFoundError:
        raise  # Re-raise for higher-level handling
    except PermissionError:
        raise
    except IsADirectoryError:
        raise
    except UnicodeDecodeError:
        raise
    except OSError as e:
        # Handle other OS-related errors
        raise OSError(f"OS error occurred while reading file: {e}") from e


def is_valid_path(path: str) -> bool:
    """
    Validate if a given path is syntactically valid.
    
    Note: This does not check if the file exists, only if the path format is valid.
    
    Args:
        path: Path string to validate
        
    Returns:
        True if the path is syntactically valid, False otherwise
    """
    # Basic path validation (platform-independent)
    if not path.strip():
        return False
        
    # Check for invalid characters (simplified version)
    invalid_chars = ['\0']  # Null character is invalid on all platforms
    if any(c in path for c in invalid_chars):
        return False
        
    # Additional platform-specific checks
    if sys.platform.startswith('win'):
        # Windows-specific invalid characters
        win_invalid = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        if any(c in path for c in win_invalid):
            return False
            
    return True


def get_valid_file_path() -> str:
    """
    Prompt the user for a file path until a valid and non-empty path is provided.
    
    Returns:
        A validated file path string
    """
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        # Prompt user for file name/path
        sys.stdout.write(f"Enter the name of the file (attempt {attempt}/{max_attempts}, or 'q' to quit): ")
        sys.stdout.flush()
        
        # Read and process input
        file_path = sys.stdin.readline().strip()
        
        # Check for quit command
        if file_path.lower() == 'q':
            sys.stderr.write("Operation cancelled by user.\n")
            sys.exit(0)
            
        # Validate path syntax
        if not is_valid_path(file_path):
            sys.stderr.write("Error: Invalid path syntax. Please enter a valid file path.\n")
            continue
            
        return file_path
        
    # If max attempts exceeded
    sys.stderr.write("Error: Maximum number of attempts reached. Exiting.\n")
    exit()


def handle_exception(e: Exception, file_path: str) -> None:
    """
    Handle exceptions in a centralized manner for better consistency.
    
    Args:
        e: The exception to handle
        file_path: Path of the file involved in the exception
    """
    error_messages = {
        FileNotFoundError: f"Error: File '{file_path}' not found",
        PermissionError: f"Error: Permission denied to read '{file_path}'",
        IsADirectoryError: f"Error: '{file_path}' is a directory, not a file",
        UnicodeDecodeError: f"Error: '{file_path}' contains non-UTF-8 data (not a text file)",
        OSError: f"OS Error: {str(e)}",
    }
    
    # Log the exception (optional, but useful for debugging)
    # logging.error(f"Exception occurred: {str(e)}", exc_info=True)
    
    # Determine the appropriate error message
    error_type = type(e)
    message = error_messages.get(error_type, f"Unexpected error: {str(e)}")
    
    # Output the error message
    sys.stderr.write(f"{message}\n")
    
    # Optionally provide suggestions based on the error type
    if error_type == FileNotFoundError:
        sys.stderr.write("Suggestion: Check if the file path is correct and the file exists.\n")
    elif error_type == PermissionError:
        sys.stderr.write("Suggestion: Ensure you have the necessary permissions to access the file.\n")
    
    # Exit with appropriate status code
    exit()


def main() -> NoReturn:
    """
    Main function to handle user input and coordinate file reading.
    
    Prompts the user for a file path, validates the input,
    and triggers the file reading process with robust error handling.
    """
    # Get a valid file path from user
    file_path = get_valid_file_path()
    
    try:
        read_and_print_file(file_path)
    except Exception as e:
        handle_exception(e, file_path)
    
    # Exit successfully if all operations complete
    exit()


if __name__ == "__main__":
    main()