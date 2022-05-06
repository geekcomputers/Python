"""
The 'cat' Program Implemented in Python 3

The Unix 'cat' utility reads the contents
of file(s) specified through stdin and 'conCATenates'
into stdout. If it is run without any filename(s) given,
then the program reads from standard input itself,
which means it simply copies stdin to stdout.

It is fairly easy to implement such a program
in Python, and as a result countless examples
exist online. This particular implementation
focuses on the basic functionality of the cat
utility. Compatible with Python 3.6 or higher.

Syntax:
python3 cat.py [filenames...]
Any number of arguments can be specified.
Separate filenames with spaces.

David Costell (DontEatThemCookies on GitHub)
v3 - 05/06/2022
"""
import sys
import argparse


def with_files(files: list) -> None:
    """
    Executes when arguments are specified.
    The argument list is passed into this
    function to be worked on.
    """
    try:
        # Read each file's contents and store them (list comprehension is used here)
        file_contents = [contents for contents in [open(file).read() for file in files]]
    except OSError as err:
        # This executes when an exception is raised (e.g. FileNotFoundError)
        print(f"cat has encountered an error!\n{err}")
        exit(1)
    else:
        # Write all file contents into the standard output stream
        for contents in file_contents:
            sys.stdout.write(contents)


def no_files() -> None:
    """
    Executes when no arguments are specified.
    Copies standard input to standard output.
    """
    try:
        # Get input, output the input, repeat
        while True:
            print(input())
    # Graceful exit for Ctrl + C, Ctrl + D
    except KeyboardInterrupt:
        exit()
    except EOFError:
        exit()


def main():
    """Entry point of the program."""
    # Read the arguments passed to the program
    parser = argparse.ArgumentParser(
        prog="cat",
        description="Simple recreation of the Unix cat utility in Python",
        epilog="version 3 - by DontEatThemCookies"
    )
    parser.add_argument("filenames", nargs="*")
    filenames = parser.parse_args().filenames

    if not filenames:
        no_files()
    else:
        with_files(filenames)


if __name__ == "__main__":
    main()
