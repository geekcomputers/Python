# Script Name		: create_dir_if_not_there.py
# Author		: Craig Richards
# Created		: 09th January 2012
# Last Modified		: 29th July 2016
# Version		: 1.1
# Modifications		: Added exceptions
#                        1.0.1 Tidy up comments and syntax
#                       : 1.1 Added User Input for Directory Name
#               	:     Added Messages with OS specific separators
#               	:     Tidied up the Indentation
# Description		: Checks to see if a directory exists in the users home directory, if not then create it

import os		# Import the OS module
dir = raw_input("Enter the directory name to be created\n")
dir = os.sep + dir

try:
    home = os.path.expanduser("~")          # Set the variable home by expanding the users set home directory
    print "Home: %s" % home                              # Print the location

    if not os.path.exists(home+dir):
        os.makedirs(home+dir)        # If not create the directory, inside their home directory
        print "Created %s" % home+dir
    else:
        print "Directory Already Present"
except OSError as e:
        print "Exception: %r" % e
