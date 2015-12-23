#!/usr/bin/python

# Script Name		: script_count.py
# Author			: Craig Richards
# Created			: 27th February 2012
# Last Modified		: 20th July 2012
# Version			: 1.3

# Modifications
#1.1 - 28-02-2012 - Changed inside github and development functions, 
#                   so instead of if os.name = "posix" do this else do this etc 
#                   I used os.path.join, so it condensed 4 lines down to 1
#1.2 - 10-05-2012 - CR - Added a line to include PHP scripts.
#1.3 - 20-07-2012 - CR - Added the line to include Batch scripts
#1.31 -23-12-2015 break into funcs to do unittest on CentOs 6.4, it might break on windows Gang Liang
# Description: This scans my scripts directory and gives a count of the different types of scripts

import sys
import os

# Set the variable path by getting the value from the OS environment variable scripts
#path = os.getenv("scripts") 	
path = os.path.dirname(os.path.realpath(__file__))
# Function to clear the screen
def clear_screen():
    cmd = clear_screen_get_cmd()
    clear_screen_to_system(cmd)

def clear_screen_get_cmd():
    cmd =''
    # Unix/Linux/MacOS/BSD/etc
    if os.name == "posix":
        cmd = 'clear'
    # DOS/Windows
    elif os.name in ("nt", "dos", "ce"):
        cmd = 'CLS'
    else:
        cmd=''
    return cmd

def clear_screen_to_system(cmd):
    if cmd:
        # Clear the Screen
        os.system(cmd)	

# count dir with extension
def count_files(path, extensions):
    counter = 0     	# Set the counter to 0

    # Loop through all the directories in the given path 
    for root, dirs, files in os.walk(path):
        for file in files:             	# For all the files
            if file.endswith(extensions):	# Count the files
                counter = counter + 1
    return counter  	# Return the count

# Start of the function just to count the files in the github directory
def github(dropbox, github):   	

    # Joins the paths to get the github directory - 1.1
    if os.path.exists(dropbox) ==True:
        print dropbox
    else:
        return

    github_dir = os.path.join(dropbox, 'github')	

    if os.path.exists(github_dir) == True:
        print github_dir
    else:
        return
    # Get a count for all the files in the directory
    github_count = sum((len(f) for _, _, f in os.walk(github_dir)))

    # If the number of files is greater then 5, then print the following messages
    if github_count > 5:	
        print '\nYou have too many in here, start uploading !!!!!'
        print 'You have: ' + str(github_count) + ' waiting to be uploaded to github!!'

    # Unless the count is 0, then print the following messages
    elif github_count == 0:	
        print '\nGithub directory is all Clear'
    else:
        # If it is any other number then print the following message, showing the number outstanding.
	    print '\nYou have: ' + str(github_count) + ' waiting to be uploaded to github!!'


# Start of the function just to count the files in the development directory
def development():   	

    # Joins the paths to get the development directory - 1.1
    dev_dir = os.path.join(path, 'development')	
    if os.path.isdir(dev_dir):
        pass
    else:
        return
    # Get a count for all the files in the directory
    dev_count = sum((len(f) for _, _, f in os.walk(dev_dir)))
    if dev_count > 10:
        # If the number of files is greater then 10, then print the following messages
        print '\nYou have too many in here, finish them or delete them !!!!!'
        print 'You have: ' + str(dev_count) + ' waiting to be finished!!'
    elif dev_count ==0:
        # Unless the count is 0, then print the following messages
        print '\nDevelopment directory is all clear'
    else:
        # If it is any other number then print the following message, showing the number outstanding.
	    print '\nYou have: ' + str(dev_count) + ' waiting to be finished!!'

def main():
    # Call the function to clear the screen
    clear_screen()	
	
    print '\nYou have the following :\n'
    count1 = count_files(path, '.py')
    # Run the count_files function to count the files with the extension we pass
    print 'Batch:\t' + str(count_files(path, ('.py', ',cmd')))	# 1.3
    print 'Perl:\t' + str(count_files(path, '.pl'))
    print 'PHP:\t' + str(count_files(path, '.php'))	# 1.2
    print 'Python:\t' + str(count_files(path, '.py'))
    print 'Shell:\t' + str(count_files(path, ('.ksh', '.sh', '.bash')))
    print 'SQL:\t' + str(count_files(path, '.sql'))

    # Set the variable dropbox by getting the value from the OS environment variable dropbox
    dropbox = os.getenv("dropbox")	
    dropbox = "~"

    # Call the github function
    github(dropbox,'python')
    development()	# Call the development function

if __name__ == "__main__":
    main()
