"""
The 'cat' Program Implemented in Python 3

The Unix 'cat' utility reads the contents
of file(s) and 'conCATenates' into stdout.
If it is run without any filename(s) given,
then the program reads from standard input,
which means it simply copies stdin to stdout.

It is fairly easy to implement such a program
in Python, and as a result countless examples
exist online. This particular implementation
focuses on the basic functionality of the cat
utility. Compatible with Python 3.6 or higher.

Syntax:
python3 cat.py [filename1 filename2 etcetera]
Separate filenames with spaces as usual.

David Costell (DontEatThemCookies on GitHub)
v1 - 02/06/2022
"""
import sys


def with_files(files):
    """Executes when file(s) is/are specified."""
    file_contents = []
    try:
        # Read the files' contents and store their contents
        for file in files:
            with open(file) as f:
                file_contents.append(f.read())
    except OSError as err:
        # This executes when there's an error (e.g. FileNotFoundError)
        print(f"cat: error reading files ({err})")

    # Write the contents of all files into the standard output stream
    for contents in file_contents:
        sys.stdout.write(contents)


def no_files():
    """Executes when no file(s) is/are specified."""
    try:
        # Loop getting input then outputting the input.
        while True:
            inp = input()
            print(inp)
    # Gracefully handle Ctrl + C and Ctrl + D
    except KeyboardInterrupt:
        exit()
    except EOFError:
        exit()


def main():
    """Entry point of the cat program."""
    try:
        # Read the arguments passed to the program
        file = sys.argv[1:]
        with_files(file)
    except IndexError:
        no_files()


if __name__ == "__main__":
    main()
