# Script Name   : env_check.py
# Author        : Craig Richards
# Created       : 14th May 2012
# Last Modified	: 14 February 2016
# Version       : 1.0.1

# Modifications	: 1.0.1 - Tidy up comments and syntax

# Description   : This script will check to see if all of the environment variables I require are set

import os

confdir = os.getenv("my_config")                # Set the variable confdir from the OS environment variable
conffile = 'env_check.conf'                     # Set the variable conffile
conffilename = os.path.join(confdir, conffile)  # Set the variable conffilename by joining confdir and conffile together

for env_check in open(conffilename):            # Open the config file and read all the settings
  env_check = env_check.strip()                 # Set the variable as itself, but strip the extra text out
  print '[{}]'.format(env_check)                # Format the Output to be in Square Brackets
  newenv = os.getenv(env_check)                 # Set the variable newenv to get the settings from the OS what is currently set for the settings out the configfile
  
  if newenv is None:                            # If it doesn't exist
    print env_check, 'is not set'               # Print it is not set
  else:                                         # Else if it does exist
    print 'Current Setting for {}={}\n'.format(env_check, newenv)	# Print out the details
