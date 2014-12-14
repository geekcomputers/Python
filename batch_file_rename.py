# Script Name		: batch_file_rename.py
# Author				: Craig Richards
# Created				: 6th August 2012
# Last Modified		: 
# Version				: 1.0

# Modifications		: 

# Description			: This will batch rename a group of files in a given directory, once you pass the current and new extensions

import os															# Load the library module
from sys import argv                            # imports argv from sys to take arguments given in the command line when the program is initiated

work_dir, old_ext, new_ext = argv[1:]       # work_dir is the work directory, old_ext and new_ext are the current and wanted file extentions respectively. 

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

