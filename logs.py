# Script Name   : logs.py
# Author        : Craig Richards
# Created       : 13th October 2011
# Last Modified	: 14 February 2016
# Version		: 1.2
#
# Modifications	: 1.1 - Added the variable zip_program so you can set it for the zip program on whichever OS, so to run on a different OS just change the locations of these two variables.
#               : 1.2 - Tidy up comments and syntax
#
# Description   : This script will search for all *.log files in the given directory, zip them using the program you specify and then date stamp them

import os  # Load the Library Module
from time import strftime  # Load just the strftime Module from Time

logsdir = "c:\puttylogs"  # Set the Variable logsdir
zip_program = "zip.exe"  # Set the Variable zip_program - 1.1

for files in os.listdir(logsdir):  # Find all the files in the directory
    if files.endswith(".log"):  # Check to ensure the files in the directory end in .log
        files1 = (
            files + "." + strftime("%Y-%m-%d") + ".zip"
        )  # Create the Variable files1, this is the files in the directory, then we add a suffix with the date and the zip extension
        os.chdir(logsdir)  # Change directory to the logsdir
        os.system(
            zip_program + " " + files1 + " " + files
        )  # Zip the logs into dated zip files for each server. - 1.1
        os.remove(files)  # Remove the original log files
