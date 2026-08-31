"""
Duplicate File Finder
---------------------
Scans a target directory to identify duplicate files by comparing their
SHA-256 checksums in memory-efficient chunks.

Features:
- Size-first filtering for fast scanning
- Memory-efficient chunked reading for large files
- Handles nested subdirectories and permission errors gracefully

Usage:
    python duplicate_file_finder.py [directory_path]
"""

import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
from typing import DefaultDict, Dict, List


def calculate_file_hash(filepath: Path, chunk_size: int = 65536) -> str:
    """
    Calculate the SHA-256 hash of a file reading in chunks.

    :param filepath: Path to the file.
    :param chunk_size: Size of byte chunks to read at a time.
    :return: Hexadecimal SHA-256 hash string.
    """
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as file_obj:
            while chunk := file_obj.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (PermissionError, OSError) as error:
        print(f"[!] Warning: Could not read {filepath}: {error}")
        return ""


def find_duplicate_files(target_dir: str) -> Dict[str, List[Path]]:
    """
    Scan a directory tree and return a dictionary mapping file hashes
    to lists of duplicate file paths.

    :param target_dir: The root directory to scan.
    :return: Dictionary of {hash: [list_of_duplicate_paths]}.
    """
    path = Path(target_dir).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {target_dir}")

    print(f"[*] Scanning directory: {path}")

    # Step 1: Group files by size (files with unique sizes cannot be duplicates)
    size_map: DefaultDict[int, List[Path]] = defaultdict(list)
    total_files = 0

    for root, _, files in os.walk(path):
        for filename in files:
            file_path = Path(root) / filename
            try:
                if not file_path.is_symlink() and file_path.exists():
                    file_size = file_path.stat().st_size
                    if file_size > 0:  # Skip empty files
                        size_map[file_size].append(file_path)
                        total_files += 1
            except (PermissionError, OSError):
                continue

    print(f"[*] Found {total_files} files. Checking for duplicate candidates...")

    # Step 2: For file sizes with multiple files, compare hashes
    hash_map: DefaultDict[str, List[Path]] = defaultdict(list)

    for size, file_list in size_map.items():
        if len(file_list) > 1:
            for file_path in file_list:
                file_hash = calculate_file_hash(file_path)
                if file_hash:
                    hash_map[file_hash].append(file_path)

    # Return only groups that have duplicates (more than 1 path)
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    return duplicates


def format_size(bytes_size: int) -> str:
    """Format bytes into a human-readable string (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find duplicate files in a directory using SHA-256 hashing."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Target directory path to scan (default: current directory)",
    )
    args = parser.parse_args()

    try:
        duplicates = find_duplicate_files(args.directory)
    except NotADirectoryError as e:
        print(f"[!] Error: {e}")
        return

    if not duplicates:
        print("\n[+] No duplicate files found.")
        return

    total_duplicate_groups = len(duplicates)
    total_wasted_space = 0

    print(f"\n{'='*60}")
    print(f"[+] Found {total_duplicate_groups} group(s) of duplicate files:")
    print(f"{'='*60}\n")

    for index, (file_hash, file_paths) in enumerate(duplicates.items(), start=1):
        file_size = file_paths[0].stat().st_size
        wasted = file_size * (len(file_paths) - 1)
        total_wasted_space += wasted

        print(f"Group #{index} (Size: {format_size(file_size)} each, SHA-256: {file_hash[:12]}...):")
        for path in file_paths:
            print(f"  -> {path}")
        print()

    print(f"{'='*60}")
    print(f"[+] Total duplicate wasted space: {format_size(total_wasted_space)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()