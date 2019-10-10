from __future__ import print_function

import os  # Load the library module

# Script Name		: script_count.py
# Author				: Craig Richards
# Created				: 27th February 2012
# Last Modified		: 20th July 2012
# Version				: 1.3
# Modifications		: 1.1 - 28-02-2012 - CR - Changed inside github and development functions, so instead of if os.name = "posix" do this else do this etc
#							: I used os.path.join, so it condensed 4 lines down to 1
#							: 1.2 - 10-05-2012 - CR - Added a line to include PHP scripts.
#							: 1.3 - 20-07-2012 - CR - Added the line to include Batch scripts
# Description			: This scans my scripts directory and gives a count of the different types of scripts

path = os.getenv("scripts")  # Set the variable path by getting the value from the OS environment variable scripts
dropbox = os.getenv("dropbox")  # Set the variable dropbox by getting the value from the OS environment variable dropbox


def clear_screen():  # Function to clear the screen
    if os.name == "posix":  # Unix/Linux/MacOS/BSD/etc
        os.system('clear')  # Clear the Screen
    elif os.name in ("nt", "dos", "ce"):  # DOS/Windows
        os.system('CLS')  # Clear the Screen


def count_files(path,
                extensions):  # Start of the function to count the files in the scripts directory, it counts the extension when passed below
    counter = 0  # Set the counter to 0
    for root, dirs, files in os.walk(path):  # Loop through all the directories in the given path
        for file in files:  # For all the files
            counter += file.endswith(extensions)  # Count the files
    return counter  # Return the count


def github():  # Start of the function just to count the files in the github directory
    github_dir = os.path.join(dropbox, 'github')  # Joins the paths to get the github directory - 1.1
    github_count = sum((len(f) for _, _, f in os.walk(github_dir)))  # Get a count for all the files in the directory
    if github_count > 5:  # If the number of files is greater then 5, then print the following messages

        print('\nYou have too many in here, start uploading !!!!!')
        print('You have: ' + str(github_count) + ' waiting to be uploaded to github!!')
    elif github_count == 0:  # Unless the count is 0, then print the following messages
        print('\nGithub directory is all Clear')
    else:  # If it is any other number then print the following message, showing the number outstanding.
        print('\nYou have: ' + str(github_count) + ' waiting to be uploaded to github!!')


def development():  # Start of the function just to count the files in the development directory
    dev_dir = os.path.join(path, 'development')  # Joins the paths to get the development directory - 1.1
    dev_count = sum((len(f) for _, _, f in os.walk(dev_dir)))  # Get a count for all the files in the directory
    if dev_count > 10:  # If the number of files is greater then 10, then print the following messages

        print('\nYou have too many in here, finish them or delete them !!!!!')
        print('You have: ' + str(dev_count) + ' waiting to be finished!!')
    elif dev_count == 0:  # Unless the count is 0, then print the following messages
        print('\nDevelopment directory is all clear')
    else:
        print('\nYou have: ' + str(
            dev_count) + ' waiting to be finished!!')  # If it is any other number then print the following message, showing the number outstanding.


clear_screen()  # Call the function to clear the screen

print('\nYou have the following :\n')
print('AutoIT:\t' + str(
    count_files(path, '.au3')))  # Run the count_files function to count the files with the extension we pass
print('Batch:\t' + str(count_files(path, ('.bat', ',cmd'))))  # 1.3
print('Perl:\t' + str(count_files(path, '.pl')))
print('PHP:\t' + str(count_files(path, '.php')))  # 1.2
print('Python:\t' + str(count_files(path, '.py')))
print('Shell:\t' + str(count_files(path, ('.ksh', '.sh', '.bash'))))
print('SQL:\t' + str(count_files(path, '.sql')))

github()  # Call the github function
development()  # Call the development function
