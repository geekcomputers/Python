# Script Name	: backup_automater_services.py
# Author			: Craig Richards
# Created			: 24th October 2012
# Last Modified	: 13th February 2016
# Version			: 1.0.1

# Modifications	: 1.0.1 - Tidy up the comments and syntax

# Description		: This will go through and backup all my automator services workflows

import datetime                         # Load the library module
import os                               # Load the library module
import shutil							# Load the library module

today    = datetime.date.today()	    # Get Today's date
todaystr = today.isoformat()		    # Format it so we can use the format to create the directory

confdir      = os.getenv("my_config")		  	   # Set the variable by getting the value from the OS setting
dropbox      = os.getenv("dropbox") 					 # Set the variable by getting the value from the OS setting
conffile     = ('services.conf') 					     # Set the variable as the name of the configuration file
conffilename = os.path.join(confdir, conffile) # Set the variable by combining the path and the file name
sourcedir    = os.path.expanduser('~/Library/Services/')	 # Source directory of where the scripts are located
destdir      = os.path.join(dropbox, "My_backups" + "/" +
    "Automater_services" + todaystr + "/")   # Combine several settings to create

                                                                                    # the destination backup directory
for file_name in open(conffilename): 									  # Walk through the configuration file
  fname = file_name.strip()													    # Strip out the blank lines from the configuration file
  if fname:																			        # For the lines that are not blank
    sourcefile = os.path.join(sourcedir, fname)		      # Get the name of the source files to backup
    destfile = os.path.join(destdir, fname) 			      # Get the name of the destination file names
    shutil.copytree(sourcefile, destfile)						  	# Copy the directories
