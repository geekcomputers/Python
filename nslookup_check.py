# Script Name		: nslookup_check.py
# Author		: Craig Richards
# Created	        : 5th January 2012
# Last Modified        : 08th April 2016
# Version              : 1.0

# Modifications        : Corrected 1st Argument of Popen

# Description	        : This very simple script opens the file server_list.txt and the does an nslookup for each one to check the DNS entry

import subprocess				           # Import the subprocess module

with open('server_list.txt') as f:
        server_list = f.read().split('\n')                 # To read server names from file and remove '\n'

if '' in server_list:                                      # Remove empty names
        server_list.remove('')

for server in server_list:
	subprocess.Popen(['nslookup', server])	           # Run the nslookup command for each server in the list
