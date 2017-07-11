# Script Name   : create_dir_if_not_there.py
# Author        : Craig Richards
# Created       : 09th January 2012
# Last Modified : 22nd October 2015
# Version       : 1.0.1
# Modifications : Added exceptions
#               : 1.0.1 Tidy up comments and syntax
#
# Description   : Checks to see if a directory exists in the users home directory, if not then create it

import os		# Import the OS module
MESSAGE = 'The directory already exists.'
TESTDIR = 'testdir'
try:
    home = os.path.expanduser("~")          # Set the variable home by expanding the user's set home directory
    print(home)                             # Print the location
    
    if not os.path.exists(os.path.join(home, TESTDIR)): # os.path.join() for making a full path safely
        os.makedirs(os.path.join(home, TESTDIR))        # If not create the directory, inside their home directory
    else:
        print(MESSAGE)
except Exception as e:
    print(e)
    
