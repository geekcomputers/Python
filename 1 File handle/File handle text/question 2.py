from typing import List, Tuple


def display_short_words(file_path: str) -> Tuple[List[str], int]:
    """
    Read a text file and extract words with fewer than 4 characters.
    
    Reads the entire content of the file using the `read()` method,
    splits it into words (whitespace-separated), filters words with length < 4,
    and returns both the filtered words and their count.
    
    Args:
        file_path: Path to the text file (e.g., "STORY.TXT")
        
    Returns:
        Tuple containing:
            - List of words with fewer than 4 characters
            - Count of such words
            
    Raises:
        FileNotFoundError: If the specified file does not exist
        PermissionError: If the user lacks permission to read the file
        IsADirectoryError: If the provided path points to a directory
        UnicodeDecodeError: If the file contains non-UTF-8 encoded data
        OSError: For other OS-related errors (e.g., invalid path)
    """
    # Read entire file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content: str = file.read()
    
    # Split content into words (handles multiple whitespace characters)
    words: List[str] = content.split()
    
    # Filter words with length < 4 using a generator expression for memory efficiency
    short_words: List[str] = [word for word in words if len(word) < 4]
    
    return short_words, len(short_words)


def main() -> None:
    """
    Main function to handle user interaction and coordinate the word extraction process.
    
    Guides the user through inputting a file path, validates inputs,
    calls the word extraction function, and displays results with error handling.
    """
    print("=== Short Word Extractor ===")
    print("This program reads a text file and displays words with fewer than 4 characters.\n")
    
    # Get and validate file path input
    while True:
        file_path: str = input("Enter the path to the text file (e.g., 'STORY.TXT'): ").strip()
        
        if not file_path:
            print("Error: File path cannot be empty. Please try again.\n")
            continue
        break
    
    try:
        # Extract short words
        short_words, count = display_short_words(file_path)
        
        # Display results
        print(f"\nFound {count} words with fewer than 4 characters:")
        if short_words:
            # Print words in chunks of 10 for readability
            for i in range(0, count, 10):
                chunk = short_words[i:i+10]
                print(" ".join(chunk))
        else:
            print("No words with fewer than 4 characters found.")
    
    except FileNotFoundError:
        print(f"\nError: File '{file_path}' not found. Please check the path.")
    except PermissionError:
        print(f"\nError: Permission denied to read '{file_path}'. Check your access rights.")
    except IsADirectoryError:
        print(f"\nError: '{file_path}' is a directory, not a file.")
    except UnicodeDecodeError:
        print(f"\nError: '{file_path}' is not a valid UTF-8 text file.")
    except OSError as e:
        print(f"\nSystem error: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        print("\nProgram completed.")


if __name__ == "__main__":
    main()
                




