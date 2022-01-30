# Script Name		: check_file.py

# Author		: Craig Richards
# Created		: 20 May 2013
# Last Modified		:
# Version		: 1.0

# Modifications	: with statement added to ensure correct file closure

# Description	: Check a file exists and that we can read the file
from __future__ import print_function

import os  # Import the Modules
import sys  # Import the Modules


# Prints usage if not appropriate length of arguments are provided


def usage():
    print("[-] Usage: python check_file.py [filename1] [filename2] ... [filenameN]")


# Readfile Functions which open the file that is passed to the script
def readfile(filename):
    with open(filename, "r") as f:  # Ensure file is correctly closed under
        read_file = f.read()  # all circumstances
    print(read_file)
    print()
    print("#" * 80)
    print()


def main():
    # Check the arguments passed to the script
    if len(sys.argv) >= 2:
        file_names = sys.argv[1:]
        filteredfilenames_1 = list(
            file_names
        )  # To counter changing in the same list which you are iterating
        filteredfilenames_2 = list(file_names)
        # Iterate for each filename passed in command line argument
        for filename in filteredfilenames_1:
            if not os.path.isfile(filename):  # Check the File exists
                print("[-] " + filename + " does not exist.")
                filteredfilenames_2.remove(
                    filename
                )  # remove non existing files from fileNames list
                continue

            # Check you can read the file
            if not os.access(filename, os.R_OK):
                print("[-] " + filename + " access denied")
                # remove non readable fileNames
                filteredfilenames_2.remove(filename)
                continue

        # Read the content of each file that both exists and is readable
        for filename in filteredfilenames_2:
            # Display Message and read the file contents
            print("[+] Reading from : " + filename)
            readfile(filename)

    else:
        usage()  # Print usage if not all parameters passed/Checked


if __name__ == "__main__":
    main()
