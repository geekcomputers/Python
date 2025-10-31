#!/usr/bin/env python3
"""
Smart File Organizer

A utility script to organize files in a specified directory into categorized
subfolders based on file types.

Example categories include: Images, Documents, Videos, Audios, Archives, Scripts, Others.

Usage:
    python smart_file_organizer.py --path "C:\\Users\\YourName\\Downloads" --interval 0

Arguments:
    --path       Directory path to organize.
    --interval   Interval in minutes to repeat automatically (0 = run once).

Author:
    Sangam Paudel
"""

import os
import shutil
import argparse
import time
from datetime import datetime

FILE_CATEGORIES = {
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg"],
    "Documents": [".pdf", ".doc", ".docx", ".txt", ".ppt", ".pptx", ".xls", ".xlsx"],
    "Videos": [".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv"],
    "Audios": [".mp3", ".wav", ".aac", ".flac", ".ogg"],
    "Archives": [".zip", ".rar", ".tar", ".gz", ".7z"],
    "Scripts": [".py", ".js", ".sh", ".bat", ".java", ".cpp", ".c"],
}


def create_folder(folder_path: str) -> None:
    """
    Create a folder if it does not already exist.

    Args:
        folder_path: Path of the folder to create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_category(file_ext: str) -> str:
    """
    Determine the category of a file based on its extension.

    Args:
        file_ext: File extension (e.g., ".txt").

    Returns:
        Category name (e.g., "Documents") or "Others" if not matched.
    """
    for category, extensions in FILE_CATEGORIES.items():
        if file_ext.lower() in extensions:
            return category
    return "Others"


def organize_files(base_path: str) -> None:
    """
    Organize files in the given directory into subfolders by category.

    Args:
        base_path: Path of the directory to organize.
    """
    files = [
        f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))
    ]
    if not files:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] No files found in {base_path}")
        return

    for file_name in files:
        source = os.path.join(base_path, file_name)
        file_ext = os.path.splitext(file_name)[1]
        category = get_category(file_ext)
        target_folder = os.path.join(base_path, category)
        create_folder(target_folder)

        try:
            shutil.move(source, os.path.join(target_folder, file_name))
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Moved: {file_name} -> {category}/"
            )
        except Exception as e:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Error moving {file_name}: {e}"
            )


def main() -> None:
    """Parse command-line arguments and execute the file organizer."""
    parser = argparse.ArgumentParser(
        description="Organize files in a directory into categorized subfolders."
    )
    parser.add_argument("--path", required=True, help="Directory path to organize.")
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Interval (in minutes) to repeat automatically (0 = run once).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Path not found: {args.path}")
        return

    print(f"Watching directory: {args.path}")
    print("Organizer started. Press Ctrl+C to stop.\n")

    try:
        while True:
            organize_files(args.path)
            if args.interval == 0:
                break
            print(f"Waiting {args.interval} minutes before next run...\n")
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        print("\nOrganizer stopped by user.")


if __name__ == "__main__":
    main()
