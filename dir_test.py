# Script Name		: dir_test.py
# Author		: Craig Richards
# Created		: 29th November 2011
# Last Modified		: 29th July 2016
# Version		: 1.1
# Modifications		: 1.1 Added User Input for Directory Name along with else case
#                       :     Added Messages and Basic Exception Handling

# Description			: Tests to see if the directory testdir exists, if not it will create the directory for you
import os  # Import the OS module


print "Current Path: %s" % os.getcwd()
dir = raw_input("Enter Directory Name to look for in current path\n")

os_sep_dir = os.getcwd() + os.sep + dir
try:
    if not os.path.exists(dir):  # Check to see if it exists
        os.makedirs(dir)  # Create the directory
        print "%s created" % (os_sep_dir)
    else:
        print "%s already exists" % (os_sep_dir)
except OSError as e:
    print "Exception: %r" % e
