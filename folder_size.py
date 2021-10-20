# Script Name   : folder_size.py
# Author        : Craig Richards
# Created       : 19th July 2012
# Last Modified	: 22 February 2016
# Version       : 1.0.1

# Modifications : Modified the Printing method and added a few comments

# Description   : This will scan the current directory and all subdirectories and display the size.

import os
import sys  # Load the library module and the sys module for the argument vector'''

try:
    directory = sys.argv[1]  # Set the variable directory to be the argument supplied by user.
except IndexError:
    sys.exit("Must provide an argument.")

dir_size = 0  # Set the size to 0
fsizedicr = {'Bytes': 1,
             'Kilobytes': float(1) / 1024,
             'Megabytes': float(1) / (1024 * 1024),
             'Gigabytes': float(1) / (1024 * 1024 * 1024)}
for (path, dirs, files) in os.walk(
        directory):  # Walk through all the directories. For each iteration, os.walk returns the folders, subfolders and files in the dir.
    for file in files:  # Get all the files
        filename = os.path.join(path, file)
        dir_size += os.path.getsize(filename)  # Add the size of each file in the root dir to get the total size.

fsizeList = [str(round(fsizedicr[key] * dir_size, 2)) + " " + key for key in fsizedicr]  # List of units

if dir_size == 0:
    print("File Empty")  # Sanity check to eliminate corner-case of empty file.
else:
    for units in sorted(fsizeList)[::-1]:  # Reverse sort list of units so smallest magnitude units print first.
        print("Folder Size: " + units)
