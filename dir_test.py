# Script Name		: dir_test.py
# Author				: Craig Richards
# Created				: 29th November 2011
# Last Modified		:
# Version				: 1.0
# Modifications		:

# Description			: Tests to see if the directory testdir exists, if not it will create the directory for you
from __future__ import print_function

import os

try:
    input = raw_input
except NameError:
    pass


def main():
    CheckDir = input("Enter the name of the directory to check : ")
    print()

    if os.path.exists(CheckDir):  # Checks if the dir exists
        print("The directory exists")
    else:
        print("No directory found for " + CheckDir)  # Output if no directory
        print()
        os.makedirs(CheckDir)  # Creates a new dir for the given name
        print("Directory created for " + CheckDir)


if __name__ == '__main__':
    main()
