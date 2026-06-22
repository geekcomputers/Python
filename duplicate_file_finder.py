import os
import hashlib


def hash_file(filepath):
    """Return SHA256 hash of file"""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(directory):
    hashes = {}
    duplicates = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                file_hash = hash_file(path)

                if file_hash in hashes:
                    duplicates.append((path, hashes[file_hash]))
                else:
                    hashes[file_hash] = path

            except Exception as e:
                print(f"Error reading {path}: {e}")

    return duplicates


if __name__ == "__main__":
    directory = input("Enter directory to scan: ")

    if not os.path.isdir(directory):
        print("Invalid directory")
        exit()

    duplicates = find_duplicates(directory)

    if not duplicates:
        print("No duplicate files found.")
    else:
        print("\nDuplicate files:")
        for dup, original in duplicates:
            print(f"{dup} == {original}")
