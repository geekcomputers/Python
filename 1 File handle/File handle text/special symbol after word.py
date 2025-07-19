from pathlib import Path


def print_words_with_asterisk(file_path: str | Path | None = None) -> None:
    """
    Reads a text file and prints each word followed by an asterisk (*) using two methods.
    Handles file paths with spaces and subdirectories.

    Args:
        file_path (str | Path, optional): Path to the text file. 
            Defaults to "1 File handle/File handle text/happy.txt" if not specified.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        IsADirectoryError: If the path points to a directory instead of a file.
        PermissionError: If read access to the file is denied.
    """
    # Set default path if not provided (handles spaces and subdirectories)
    if file_path is None:
        file_path = Path("1 File handle/File handle text/happy.txt")
    else:
        file_path = Path(file_path)  # Convert to Path object for consistent handling

    # Validate the file path
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path.resolve()}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {file_path.resolve()}")

    try:
        with open(file_path, encoding='utf-8') as file:
            print(f"Processing file: {file_path.resolve()}\n")  # Show absolute path for verification

            # Method 1: Split entire file content into words
            content = file.read()
            words = content.split()
            
            print("Method 1 Output:")
            for word in words:
                print(f"{word}*", end="")
            print("\n")  # Newline after method 1 output

            # Reset file pointer to start for method 2
            file.seek(0)

            # Method 2: Process line by line, then split lines into words
            print("Method 2 Output:")
            for line in file:
                line_words = line.split()
                for word in line_words:
                    print(f"{word}*", end="")
            print()  # Final newline after all output

    except PermissionError:
        raise PermissionError(f"Permission denied: Cannot read {file_path.resolve()}")

if __name__ == "__main__":
    try:
        # You can explicitly pass the path if needed, e.g.:
        # custom_path = r"C:\Full\Path\To\1 File handle\File handle text\happy.txt"  # Windows
        # print_words_with_asterisk(custom_path)
        
        # Use default path (works for relative paths)
        print_words_with_asterisk()
    except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")