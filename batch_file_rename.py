import argparse
import sys
from pathlib import Path


def batch_rename(work_dir, old_ext, new_ext, dry_run=False):
    """
    Batch rename files in a directory from one extension to another.
    
    Args:
        work_dir (str): Path to the target directory.
        old_ext (str): Extension to find (e.g., '.txt' or '.tar.gz').
        new_ext (str): New extension to apply (e.g., '.md').
        dry_run (bool): If True, preview changes without applying.
    """
    work_path = Path(work_dir)
    if not work_path.is_dir():
        print(f"Error: {work_dir} is not a valid directory.", file=sys.stderr)
        return

    # Normalize extensions for library safety
    if not old_ext.startswith("."):
        old_ext = "." + old_ext
    if not new_ext.startswith("."):
        new_ext = "." + new_ext

    print(f"[*] Scanning {work_dir} for files with extension '{old_ext}'...")
    
    found_files = list(work_path.glob(f"*{old_ext}"))
    if not found_files:
        print(f"[!] No files found with extension '{old_ext}'.")
        return

    for file_path in found_files:
        # Handle compound extensions by checking ends-with
        if file_path.name.endswith(old_ext):
            new_name = file_path.name[:-len(old_ext)] + new_ext
            new_file_path = file_path.with_name(new_name)
        else:
            new_file_path = file_path.with_suffix(new_ext)
        
        if new_file_path.exists():
            print(f"[!] Skip: {new_file_path.name} already exists. Cannot rename {file_path.name}.", file=sys.stderr)
            continue

        if dry_run:
            print(f"[DRY-RUN] Would rename: {file_path.name} -> {new_file_path.name}")
        else:
            try:
                file_path.rename(new_file_path)
                print(f"[+] Renamed: {file_path.name} -> {new_file_path.name}")
            except OSError as e:
                print(f"[!] Error renaming {file_path.name}: {e}", file=sys.stderr)
                continue

    if not dry_run:
        print("[*] Batch rename completed.")


def get_parser():
    """
    Create and return the argument parser for the CLI.
    """
    parser = argparse.ArgumentParser(
        description="Change extension of files in a working directory"
    )
    parser.add_argument(
        "work_dir",
        help="The directory where to change extension",
    )
    parser.add_argument(
        "old_ext", help="Old extension (e.g., .txt or txt)"
    )
    parser.add_argument(
        "new_ext", help="New extension (e.g., .md or md)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them",
    )
    return parser


def main():
    """
    This will be called if the script is directly invoked.
    """
    parser = get_parser()
    args = parser.parse_args()

    work_dir = args.work_dir
    old_ext = args.old_ext if args.old_ext.startswith(".") else "." + args.old_ext
    new_ext = args.new_ext if args.new_ext.startswith(".") else "." + args.new_ext

    batch_rename(work_dir, old_ext, new_ext, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
