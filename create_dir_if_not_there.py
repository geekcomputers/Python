# Script Name   : create_dir_if_not_there.py
# Author        : Craig Richards
# Created       : 09th January 2012
# Last Modified : 06th April 2016
# Version       : 1.0.1
# Modifications : Added exceptions
#               : 1.0.1 Tidy up comments and syntax
#				: Corrected exceptions
#               : Added else Condition for path checking
#
# Description   : Checks to see if a directory exists in the users home directory, if not then create it

import os		# Import the OS module

try:
    home = os.path.expanduser("~")          # Set the variable home by expanding the users set home directory
    print home                              # Print the location
    
    if not os.path.exists(home+'/testdir'):
        os.makedirs(home+'/testdir')        # If not create the directory, inside their home directory
    else:
    	print 'Given directory already exists in the path'
 
except Exception as e:                      # OSError
    print e
