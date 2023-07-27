import os
import argparse
from typing import Union

def generate_unique_name(directory: str, name: str) -> str:
    """
    Generate a unique name for a file or folder in the specified directory.

    Parameters:
    ----------
    directory : str
        The path to the directory.
    name : str
        The name of the file or folder.

    Returns:
    -------
    str
        The unique name with an index.
    """
    base_name, extension = os.path.splitext(name)
    index = 1
    while os.path.exists(os.path.join(directory, f"{base_name}_{index}{extension}")):
        index += 1
    return f"{base_name}_{index}{extension}"

def rename_files_and_folders(directory: str) -> None:
    """
    Rename files and folders in the specified directory to lowercase with underscores.

    Parameters:
    ----------
    directory : str
        The path to the directory containing the files and folders to be renamed.

    Returns:
    -------
    None
    """
    if not os.path.isdir(directory):
        raise ValueError("Invalid directory path.")

    for name in os.listdir(directory):
        old_path = os.path.join(directory, name)
        new_name = name.lower().replace(" ", "_")
        new_path = os.path.join(directory, new_name)

        # Check if the new filename is different from the old filename
        if new_name != name:
            # Check if the new filename already exists in the directory
            if os.path.exists(new_path):
                # If the new filename exists, generate a unique name with an index
                new_path = generate_unique_name(directory, new_name)

            os.rename(old_path, new_path)

def main() -> None:
    """
    Main function to handle command-line arguments and call the renaming function.

    Usage:
    ------
    python script_name.py <directory_path>

    Example:
    --------
    python rename_files_script.py /path/to/directory

    """
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description="Rename files and folders to lowercase with underscores.")
    parser.add_argument("directory", type=str, help="Path to the directory containing the files and folders to be renamed.")
    args = parser.parse_args()

    # Call the rename_files_and_folders function with the provided directory path
    rename_files_and_folders(args.directory)

if __name__ == "__main__":
    main()
