# Script Name		: folder_size.py
# Author				: Craig Richards
# Created				: 19th July 2012
# Last Modified		: 
# Version				: 1.0

# Modifications		: 

# Description			: This will scan the current directory and all subdirectories and display the size.

import os 														# Load the library module

directory = '.'													# Set the variable directory to be the current directory
dir_size = 0													# Set the size to 0
for (path, dirs, files) in os.walk(directory):			# Walk through all the directories
  for file in files:												# Get all the files
    filename = os.path.join(path, file)
    dir_size += os.path.getsize(filename)			# Get the sizes, the following lines print the sizes in bytes, Kb, Mb and Gb
print "Folder Size in Bytes = %0.2f Bytes" % (dir_size)
print "Folder Size in Kilobytes = %0.2f KB" % (dir_size/1024.0)
print "Folder Size in Megabytes = %0.2f MB" % (dir_size/1024/1024.0)
print "Folder Size in Gigabytes = %0.2f GB" % (dir_size/1024/1024/1024.0)