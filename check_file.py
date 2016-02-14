# Script Name	: check_file.py
# Author		: Craig Richards
# Created		: 20 May 2013 
# Last Modified	: 
# Version		: 1.0

# Modifications	: 

# Description	: Check a file exists and that we can read the file

import sys		# Import the Modules
import os		# Import the Modules

# Readfile Functions which open the file that is passed to the script

def readfile(filename):
	line = open(filename, 'r').read()
	print line

def main():
  if len(sys.argv) == 2:		# Check the arguments passed to the script
    filename = sys.argv[1]		# The filename is the first argument
    if not os.path.isfile(filename):	# Check the File exists
      print '[-] ' + filename + ' does not exist.'
      exit(0)
    if not os.access(filename, os.R_OK):	# Check you can read the file
      print '[-] ' + filename + ' access denied'
      exit(0)
  else:
    print '[-] Usage: ' + str(sys.argv[0]) + ' <filename>' # Print usage if not all parameters passed/Checked
    exit(0)
  print '[+] Reading from : ' + filename	# Display Message and read the file contents
  readfile(filename)
  
if __name__ == '__main__':
  main()
