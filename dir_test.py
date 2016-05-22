# Script Name		: dir_test.py
# Author				: Craig Richards
# Created				: 29th November 2011
# Last Modified		:
# Version				: 1.0
# Modifications		: 

# Description			: Tests to see if the directory testdir exists, if not it will create the directory for you

import os  # Import the OS module

if not os.path.exists('testdir'):  # Check to see if it exists
    os.makedirs('testdir')  # Create the directory
