# Script Name		: dir_test.py
# Author				: Craig Richards
# Created				: 29th November 2011
# Last Modified		:
# Version				: 1.0
# Modifications		: 

# Description			: Tests to see if the directory testdir exists, if not it will create the directory for you

import os									# Import the OS module
dir = 'testdir'
def main(dir):
    mkdir_python(dir)

def mkdir_python(dir):
    if not os.path.exists(dir):		#  Check to see if it exists  
        os.makedirs(dir)			#  Create the directory 

if __name__ == '__main__':
    main(dir)
