#!/usr/bin/env python3
"""
Author: Ivan Costa Neto
Date: 13-01-26

Auto-rename photos by timestamp, so you can organize those vacation trip photos!!

Name format: YYYY-MM-DD_HH-MM-SS[_NN].ext

Uses EXIF DateTimeOriginal when available (best for JPEG),
otherwise falls back to file modified time,

i.e.
  python rename_photos.py ~/Pictures/Trip --dry-run
  python rename_photos.py ~/Pictures/Trip --recursive
  python rename_photos.py . --prefix Japan --recursive
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}

# EXIF support is optional (w\ Pillow)
try:
    from PIL import Image, ExifTags  # type: ignore
    PIL_OK = True
except Exception:
    PIL_OK = False


def is_photo(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS


def sanitize_prefix(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    s = re.sub(r"[^\w\-]+", "_", s)
    return s[:50]


def exif_datetime_original(path: Path) -> datetime | None:
    """
    Try to read EXIF DateTimeOriginal/DateTime from image.
    Returns None if unavailable.
    """
    if not PIL_OK:
        return None
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None

        # map EXIF tag ids -> names
        tag_map = {}
        for k, v in ExifTags.TAGS.items():
            tag_map[k] = v

        # common EXIF datetime tags
        dto = None
        dt = None
        for tag_id, value in exif.items():
            name = tag_map.get(tag_id)
            if name == "DateTimeOriginal":
                dto = value
            elif name == "DateTime":
                dt = value

        raw = dto or dt
        if not raw:
            return None

        # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
        raw = str(raw).strip()
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def file_mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def unique_name(dest_dir: Path, base: str, ext: str) -> Path:
    """
    If base.ext exists, append _01, _02, ...
    """
    cand = dest_dir / f"{base}{ext}"
    if not cand.exists():
        return cand
    i = 1
    while True:
        cand = dest_dir / f"{base}_{i:02d}{ext}"
        if not cand.exists():
            return cand
        i += 1


@dataclass
class Options:
    folder: Path
    recursive: bool
    dry_run: bool
    prefix: str
    keep_original: bool  # if true, don't rename if it already matches our format


def already_formatted(name: str) -> bool:
    # matches: YYYY-MM-DD_HH-MM-SS or with prefix and/or _NN
    pattern = r"^(?:[A-Za-z0-9_]+_)?\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d{2})?$"
    return re.match(pattern, Path(name).stem) is not None


def gather_photos(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        return [p for p in folder.rglob("*") if is_photo(p)]
    return [p for p in folder.iterdir() if is_photo(p)]


def rename_photos(opts: Options) -> int:
    photos = gather_photos(opts.folder, opts.recursive)
    photos.sort()

    if not photos:
        print("No supported photo files found.")
        return 0

    if opts.prefix:
        pref = sanitize_prefix(opts.prefix)
    else:
        pref = ""

    renamed = 0
    for p in photos:
        if opts.keep_original and already_formatted(p.name):
            continue

        dt = exif_datetime_original(p) or file_mtime(p)
        base = dt.strftime("%Y-%m-%d_%H-%M-%S")
        if pref:
            base = f"{pref}_{base}"

        dest = unique_name(p.parent, base, p.suffix.lower())

        if dest.name == p.name:
            continue

        if opts.dry_run:
            print(f"[DRY] {p.relative_to(opts.folder)} -> {dest.name}")
        else:
            p.rename(dest)
            print(f"[OK ] {p.relative_to(opts.folder)} -> {dest.name}")
            renamed += 1

    if not opts.dry_run:
        print(f"\nDone. Renamed {renamed} file(s).")
    return renamed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Auto-rename photos using EXIF date (or file modified time).")
    ap.add_argument("folder", help="Folder containing photos")
    ap.add_argument("--recursive", action="store_true", help="Process subfolders too")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without renaming")
    ap.add_argument("--prefix", default="", help="Optional prefix (e.g., Japan, RWTH, Trip)")
    ap.add_argument("--keep-original", action="store_true",
                    help="Skip files that already match YYYY-MM-DD_HH-MM-SS naming")
    args = ap.parse_args(argv)

    folder = Path(args.folder).expanduser()
    if not folder.exists() or not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 2

    if not PIL_OK:
        print("[Note] Pillow not installed; EXIF dates won't be read (mtime fallback only).")
        print("       Install for best results: pip install pillow")

    opts = Options(
        folder=folder,
        recursive=args.recursive,
        dry_run=args.dry_run,
        prefix=args.prefix,
        keep_original=args.keep_original,
    )
    rename_photos(opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
