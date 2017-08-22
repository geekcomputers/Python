# Script Name		: script_listing.py
# Author				: Craig Richards
# Created				: 15th February 2012
# Last Modified		: 29th May 2012
# Version				: 1.2

# Modifications		: 1.1 - 28-02-2012 - CR - Added the variable to get the logs directory, I then joined the output so the file goes to the logs directory
#							: 1.2 - 29-05/2012 - CR - Changed the line so it doesn't ask for a directory, it now uses the environment varaible scripts

# Description			: This will list all the files in the given directory, it will also go through all the subdirectories as well

import os																		# Load the library module							

logdir  = os.getenv("logs")												# Set the variable logdir by getting the value from the OS environment variable logs
logfile = 'script_list.log'													# Set the variable logfile
path    = os.getenv("scripts")												# Set the varable path by getting the value from the OS environment variable scripts - 1.2

#path = (raw_input("Enter dir: "))										  # Ask the user for the directory to scan
logfilename = os.path.join(logdir, logfile)					  	# Set the variable logfilename by joining logdir and logfile together
log = open(logfilename, 'w')												    # Set the variable log and open the logfile for writing

for dirpath, dirname, filenames in os.walk(path):				# Go through the directories and the subdirectories
  for filename in filenames:											    	# Get all the filenames
    log.write(os.path.join(dirpath, filename)+'\n')					# Write the full path out to the logfile

print ("\nYour logfile " , logfilename, "has been created")		# Small message informing the user the file has been created
