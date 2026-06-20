"""
A simple Python implementation of the Unix cat command.

Author:
- Nitkarsh Chourasia

Features:
- Reads one or more files
- Reads from stdin when no files are given
- Supports "-" as stdin
- Prints errors to stderr
- Continues after file errors
- Uses proper exit codes
- Supports:
  -n : number all lines
  -b : number non-empty lines
  -s : squeeze repeated blank lines
  -E : show $ at end of each line

Design notes:
- Files and stdin are both handled as streams.
- Line numbering state is shared across files, matching cat-style behavior.
- File errors are reported to stderr while processing continues.
"""

import argparse
import sys

__author__ = "Nitkarsh Chourasia"


def process_stream(stream, args, state):
    """Read from a stream and write processed output to stdout."""
    for line in stream:
        is_blank = line == "\n"

        if args.squeeze_blank and is_blank and state["previous_was_blank"]:
            continue

        state["previous_was_blank"] = is_blank

        if args.show_ends:
            if line.endswith("\n"):
                line = line[:-1] + "$\n"
            else:
                line = line + "$"

        if args.number_nonblank:
            if not is_blank:
                sys.stdout.write(f"{state['line_number']:6}\t")
                state["line_number"] += 1
        elif args.number:
            sys.stdout.write(f"{state['line_number']:6}\t")
            state["line_number"] += 1

        sys.stdout.write(line)


def process_file(filename, args, state):
    """Open one file and process its contents."""
    with open(filename, "r", encoding="utf-8") as file:
        process_stream(file, args, state)


def process_files(files, args):
    """Process all given filenames."""
    had_error = False

    state = {
        "line_number": 1,
        "previous_was_blank": False,
    }

    for filename in files:
        try:
            if filename == "-":
                process_stream(sys.stdin, args, state)
            else:
                process_file(filename, args, state)
        except OSError as err:
            print(f"cat: {filename}: {err}", file=sys.stderr)
            had_error = True

    return had_error


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="A simple Python cat command.")

    parser.add_argument(
        "files",
        nargs="*",
        help="Files to read. Use '-' to read from standard input.",
    )

    parser.add_argument(
        "-n",
        "--number",
        action="store_true",
        help="Number all output lines.",
    )

    parser.add_argument(
        "-b",
        "--number-nonblank",
        action="store_true",
        help="Number non-empty output lines.",
    )

    parser.add_argument(
        "-s",
        "--squeeze-blank",
        action="store_true",
        help="Suppress repeated empty output lines.",
    )

    parser.add_argument(
        "-E",
        "--show-ends",
        action="store_true",
        help="Display $ at the end of each line.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.files:
        state = {
            "line_number": 1,
            "previous_was_blank": False,
        }
        process_stream(sys.stdin, args, state)
        sys.exit(0)

    had_error = process_files(args.files, args)

    if had_error:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
