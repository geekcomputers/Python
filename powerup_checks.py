from __future__ import print_function

import os  # Load the Library Module
import sqlite3  # Load the Library Module
import subprocess  # Load the Library Module
import sys  # Load the Library Module
from time import strftime  # Load just the strftime Module from Time

# Script Name		: powerup_checks.py
# Author				: Craig Richards
# Created				: 25th June 2013
# Last Modified		:
# Version				: 1.0
# Modifications		:
# Description			: Creates an output file by pulling all the servers for the given site from SQLITE database, then goes through the list pinging the servers to see if they are up on the network

dropbox = os.getenv("dropbox")  # Set the variable, by getting the value of the variable from the OS
config = os.getenv("my_config")  # Set the variable, by getting the value of the variable from the OS
dbfile = ("Databases/jarvis.db")  # Set the variable to the database
master_db = os.path.join(dropbox, dbfile)  # Create the variable by linking the path and the file
listfile = ("startup_list.txt")  # File that will hold the servers
serverfile = os.path.join(config, listfile)  # Create the variable by linking the path and the file
outputfile = ('server_startup_' + strftime("%Y-%m-%d-%H-%M") + '.log')

# Below is the help text

text = '''

You need to pass an argument, the options the script expects is 

    -site1		For the Servers relating to site1
    -site2	For the Servers located in site2'''


def windows():  # This is the function to run if it detects the OS is windows.
    f = open(outputfile, 'a')  # Open the logfile
    for server in open(serverfile, 'r'):  # Read the list of servers from the list
        # ret = subprocess.call("ping -n 3 %s" % server.strip(), shell=True,stdout=open('NUL', 'w'),stderr=subprocess.STDOUT)	# Ping the servers in turn
        ret = subprocess.call("ping -n 3 %s" % server.strip(), stdout=open('NUL', 'w'),
                              stderr=subprocess.STDOUT)  # Ping the servers in turn
        if ret == 0:  # Depending on the response
            f.write("%s: is alive" % server.strip().ljust(15) + "\n")  # Write out to the logfile is the server is up
        else:
            f.write(
                "%s: did not respond" % server.strip().ljust(15) + "\n")  # Write to the logfile if the server is down


def linux():  # This is the function to run if it detects the OS is nix.
    f = open('server_startup_' + strftime("%Y-%m-%d") + '.log', 'a')  # Open the logfile
    for server in open(serverfile, 'r'):  # Read the list of servers from the list
        ret = subprocess.call("ping -c 3 %s" % server, shell=True, stdout=open('/dev/null', 'w'),
                              stderr=subprocess.STDOUT)  # Ping the servers in turn
        if ret == 0:  # Depending on the response
            f.write("%s: is alive" % server.strip().ljust(15) + "\n")  # Write out to the logfile is the server is up
        else:
            f.write(
                "%s: did not respond" % server.strip().ljust(15) + "\n")  # Write to the logfile if the server is down


def get_servers(query):  # Function to get the servers from the database
    conn = sqlite3.connect(master_db)  # Connect to the database
    cursor = conn.cursor()  # Create the cursor
    cursor.execute('select hostname from tp_servers where location =?', (query,))  # SQL Statement
    print('\nDisplaying Servers for : ' + query + '\n')
    while True:  # While there are results
        row = cursor.fetchone()  # Return the results
        if row == None:
            break
        f = open(serverfile, 'a')  # Open the serverfile
        f.write("%s\n" % str(row[0]))  # Write the server out to the file
        print(row[0])  # Display the server to the screen
        f.close()  # Close the file


def main():  # Main Function
    if os.path.exists(serverfile):  # Checks to see if there is an existing server file
        os.remove(serverfile)  # If so remove it

    if len(sys.argv) < 2:  # Check there is an argument being passed
        print(text)  # Display the help text if there isn't one passed
        sys.exit()  # Exit the script

    if '-h' in sys.argv or '--h' in sys.argv or '-help' in sys.argv or '--help' in sys.argv:  # If the ask for help
        print(text)  # Display the help text if there isn't one passed
        sys.exit(0)  # Exit the script after displaying help
    else:
        if sys.argv[1].lower().startswith('-site1'):  # If the argument is site1
            query = 'site1'  # Set the variable to have the value site
        elif sys.argv[1].lower().startswith('-site2'):  # Else if the variable is bromley
            query = 'site2'  # Set the variable to have the value bromley
        else:
            print('\n[-] Unknown option [-] ' + text)  # If an unknown option is passed, let the user know
            sys.exit(0)
    get_servers(query)  # Call the get servers funtion, with the value from the argument

    if os.name == "posix":  # If the OS is linux.
        linux()  # Call the linux function
    elif os.name in ("nt", "dos", "ce"):  # If the OS is Windows...
        windows()  # Call the windows function

    print('\n[+] Check the log file ' + outputfile + ' [+]\n')  # Display the name of the log


if __name__ == '__main__':
    main()  # Call the main function
