# Script Name   : folder_size.py
# Author        : Craig Richards (Simplified by Assistant)
# Created       : 19th July 2012
# Last Modified : 19th December 2024
# Version       : 2.0.0

# Description   : Scans a directory and subdirectories to display the total size.

import os
import sys

def get_folder_size(directory):
    """Calculate the total size of a directory and its subdirectories."""
    total_size = 0
    for root, _, files in os.walk(directory):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    return total_size

def format_size(size):
    """Format the size into human-readable units."""
    units = ["Bytes", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

def main():
    if len(sys.argv) < 2:
        print("Usage: python folder_size.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        sys.exit(1)
    
    folder_size = get_folder_size(directory)
    print(f"Folder Size: {format_size(folder_size)}")

if __name__ == "__main__":
    main()
