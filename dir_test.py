"""
Tests to see if the directory testdir exists and if not
it will create the directory for you.
"""
# Script Name		: dir_test.py
# Author				: Craig Richards
# Created				: 29th November 2011
# Last Modified		:
# Version				: 1.0
# Modifications		:

# Description			: Tests to see if the directory testdir exists,
# if not it will create the directory for you
from __future__ import print_function
import os  # Import the OS Module
import sys


def main():
    """
    Main function for the script.
    """
    if sys.version_info.major >= 3:  # if the interpreter version is 3.X,
        input_func = input           # use input, otherwise use 'raw_input'
    else:
        input_func = raw_input

    check_dir = input_func("Enter the name of the directory to check : ")
    print()

    if os.path.exists(check_dir):  # Checks if the dir exists
        print("The directory exists")
    else:
        print("No directory found for " + check_dir)  # Output if no directory
        print()
        os.makedirs(check_dir)  # Creates a new dir for the given name
        print("Directory created for " + check_dir)


if __name__ == '__main__':
    main()
