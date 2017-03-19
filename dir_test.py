# Script Name		: dir_test.py
# Author				: Craig Richards
# Created				: 29th November 2011
# Last Modified		:
# Version				: 1.0
# Modifications		:

# Description			: Tests to see if the directory testdir exists, if not it will create the directory for you

import os    # Import the OS module
DirCheck = raw_input("Please enter directory name to check : ")
print
print "There was no directory under the name " +DirCheck
print
print "So, a new directory under the name " +DirCheck + " has been created!"
if not os.path.exists(DirCheck):  # Check to see if it exists
    os.makedirs(DirCheck)  # Create the directory
