"""
Code to manage directories in the project's parent directory.

Note: Uses pathlib for platform-agnostic path handling.
All operations default to the project's parent directory.
"""

from pathlib import Path
from shutil import copytree, move

# Get the project's parent directory
PROJECT_ROOT = Path(__file__).parent.parent


def create_directory(name: str) -> None:
    """Create a directory in the project's parent directory."""
    target_dir = PROJECT_ROOT / name
    if target_dir.exists():
        print(f"Error: Folder '{name}' already exists.")
    else:
        target_dir.mkdir(exist_ok=False)
        print(f"Created directory: {target_dir}")


def delete_directory(name: str) -> None:
    """Delete a directory and its contents recursively."""
    target_dir = PROJECT_ROOT / name
    if not target_dir.exists():
        print(f"Error: Directory '{name}' does not exist.")
        return
    # Safely delete non-empty directory
    for item in target_dir.rglob("*"):
        if item.is_file():
            item.unlink()
        else:
            item.rmdir()
    target_dir.rmdir()
    print(f"Deleted directory: {target_dir}")


def rename_directory(old_name: str, new_name: str) -> None:
    """Rename a directory in the project's parent directory."""
    old_path = PROJECT_ROOT / old_name
    new_path = PROJECT_ROOT / new_name

    if not old_path.exists():
        print(f"Error: Directory '{old_name}' does not exist.")
        return
    if new_path.exists():
        print(f"Error: Directory '{new_name}' already exists.")
        return

    old_path.rename(new_path)
    print(f"Renamed '{old_name}' to '{new_name}'")


def backup_files(drive_letter: str, backup_folder: str) -> None:
    """Backup the project directory to another drive."""
    source = PROJECT_ROOT
    destination = Path(f"{drive_letter}:/{backup_folder}")

    if destination.exists():
        print(f"Error: Backup directory '{destination}' already exists.")
        return

    copytree(source, destination)
    print(f"Backed up project to: {destination}")


def move_folder(file_or_folder: str, drive_letter: str, target_folder: str) -> None:
    """Move a file/folder to a specific location, overwriting if it exists."""
    source = PROJECT_ROOT / file_or_folder
    destination = Path(f"{drive_letter}:/{target_folder}")

    if not source.exists():
        print(f"Error: Source '{source}' does not exist.")
        return

    destination.mkdir(parents=True, exist_ok=True)
    move(str(source), str(destination / source.name))
    print(f"Moved '{source}' to '{destination}'")


# Example usage
if __name__ == "__main__":
    # create_directory("test")
    # rename_directory("test", "demo")
    # delete_directory("demo")
    # backup_files('D', 'backup_project')
    # move_folder('test.txt', 'D', 'name')
    pass
