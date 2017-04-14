# Script Name		: check_file.py

# Author		: Craig Richards
# Created		: 20 May 2013
# Last Modified		:
# Version		: 1.0

# Modifications	: with statement added to ensure correct file closure

# Description	: Check a file exists and that we can read the file
from __future__ import print_function
import sys		# Import the Modules
import os		# Import the Modules

# Prints usage if not appropriate length of arguments are provided
def usage():
    print('[-] Usage: python check_file.py <filename1> [filename2] ... [filenameN]')
    exit(0)


# Readfile Functions which open the file that is passed to the script
def readfile(filename):
	with open(filename, 'r') as f:      # Ensure file is correctly closed under all circumstances
	    file = f.read()
	print(file)

def main():
  if len(sys.argv) >= 2:		# Check the arguments passed to the script
      filenames = sys.argv[1:]
      for filename in filenames: 				# Iterate for each filename passed in command line argument
          if not os.path.isfile(filename):			# Check the File exists
              print ('[-] ' + filename + ' does not exist.')
              filenames.remove(filename)			#remove non existing files from filenames list
              continue

          if not os.access(filename, os.R_OK):	# Check you can read the file
              print ('[-] ' + filename + ' access denied')
              filenames.remove(filename)			# remove non readable filenames
              continue
  else:
    usage() # Print usage if not all parameters passed/Checked

    # Read the content of each file
  for filename in filenames:
      print ('[+] Reading from : ' + filename)	# Display Message and read the file contents
      readfile(filename)

if __name__ == '__main__':
    main()
