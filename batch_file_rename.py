# Script Name		: batch_file_rename.py
# Author				: Craig Richards
# Created				: 6th August 2012
# Last Modified		: 
# Version				: 1.0

# Modifications		: 

# Description			: This will batch rename a group of files in a given directory, once you pass the current and new extensions

import os															# Load the library module
#import sys															# Load the library module
from sys import argv

#work_dir=sys.argv[1]											# Set the variable work_dir with the first argument passed
#old_ext=sys.argv[2]											# Set the variable work_dir with the first argument passed
#new_ext=sys.argv[3]											# Set the variable work_dir with the first argument passed
script, work_dir, old_ext, new_ext = argv

files = os.listdir(work_dir)									# Set the variable files, by listing everything in the directory 
for filename in files:											# Loop through the files
  file_ext = os.path.splitext(filename)[1]				# Get the file extension
  if old_ext == file_ext:										# Start of the logic to check the file extensions, if old_ext = file_ext
    newfile = filename.replace(old_ext, new_ext)	# Set newfile to be the filename, replaced with the new extension
    while os.path.isfile(newfile):                         # Check if the new file name already exists to prevent overwrite
        num = 2                                             # a number to add to the file name
        num_new_ext = "_" + str(num) + new_ext              # creates a new string to append to the end of file name
        newfile = filename.replace(old_ext, num_new_ext)    # apply changes to the new file name
        num += 1                                            # increment the number in case the name still exists
    os.rename(													# Write the files
	    os.path.join(work_dir, filename),
		os.path.join(work_dir, newfile))
