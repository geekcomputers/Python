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
python3 cat.py [filename1] [filename2] etc...
Separate filenames with spaces.

David Costell (DontEatThemCookies on GitHub)
v2 - 03/12/2022
"""
import sys

def with_files(files):
    """Executes when file(s) is/are specified."""
    try:
        # Read each file's contents and store them
        file_contents = [contents for contents in [open(file).read() for file in files]]
    except OSError as err:
        # This executes when there's an error (e.g. FileNotFoundError)
        exit(print(f"cat: error reading files ({err})"))

    # Write all file contents into the standard output stream
    for contents in file_contents:
        sys.stdout.write(contents)

def no_files():
    """Executes when no file(s) is/are specified."""
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
    """Entry point of the cat program."""
    # Read the arguments passed to the program
    if not sys.argv[1:]:
        no_files()
    else:
        with_files(sys.argv[1:])

if __name__ == "__main__":
    main()
