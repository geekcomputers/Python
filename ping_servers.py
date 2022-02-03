from __future__ import print_function

import os  # Load the Library Module
import subprocess  # Load the Library Module
import sys  # Load the Library Module

# Script Name		: ping_servers.py
# Author				: Craig Richards
# Created				: 9th May 2012
# Last Modified		: 14th May 2012
# Version				: 1.1
# Modifications		: 1.1 - 14th May 2012 - CR Changed it to use the config directory to store the server files
# Description	 : This script will, depending on the arguments supplied will ping the
# servers associated with that application group.

filename = sys.argv[0]  # Sets a variable for the script name
if (
    "-h" in sys.argv or "--h" in sys.argv or "-help" in sys.argv or "--help" in sys.argv
):  # Help Menu if called
    print(
        """
You need to supply the application group for the servers you want to ping, i.e.
    dms
    swaps

Followed by the site i.e.
    155
    bromley"""
    )
    sys.exit(0)
else:

    if (
        len(sys.argv) < 3
    ):  # If no arguments are passed,display the help/instructions on how to run the script
        sys.exit(
            "\nYou need to supply the app group. Usage : "
            + filename
            + " followed by the application group i.e. \n \t dms or \n \t swaps \n "
            "then the site i.e. \n \t 155 or \n \t bromley"
        )

    appgroup = sys.argv[1]  # Set the variable appgroup as the first argument you supply
    site = sys.argv[2]  # Set the variable site as the second argument you supply

    if os.name == "posix":  # Check the os, if it's linux then
        myping = "ping -c 2 "  # This is the ping command
    elif os.name in ("nt", "dos", "ce"):  # Check the os, if it's windows then
        myping = "ping -n 2 "  # This is the ping command

    if "dms" in sys.argv:  # If the argument passed is dms then
        appgroup = "dms"  # Set the variable appgroup to dms
    elif "swaps" in sys.argv:  # Else if the argment passed is swaps then
        appgroup = "swaps"  # Set the variable appgroup to swaps

    if "155" in sys.argv:  # If the argument passed is 155 then
        site = "155"  # Set the variable site to 155
    elif "bromley" in sys.argv:  # Else if the argument passed is bromley
        site = "bromley"  # Set the variable site to bromley

logdir = os.getenv("logs")  # Set the variable logdir by getting the OS environment logs
logfile = (
    "ping_" + appgroup + "_" + site + ".log"
)  # Set the variable logfile, using the arguments passed to create the logfile
logfilename = os.path.join(
    logdir, logfile
)  # Set the variable logfilename by joining logdir and logfile together
confdir = os.getenv(
    "my_config"
)  # Set the variable confdir from the OS environment variable - 1.2
conffile = appgroup + "_servers_" + site + ".txt"  # Set the variable conffile - 1.2
conffilename = os.path.join(
    confdir, conffile
)  # Set the variable conffilename by joining confdir and conffile together - 1.2

f = open(logfilename, "w")  # Open a logfile to write out the output
for server in open(conffilename):  # Open the config file and read each line - 1.2
    ret = subprocess.call(
        myping + server, shell=True, stdout=f, stderr=subprocess.STDOUT
    )  # Run the ping command for each server in the list.
    if ret == 0:  # Depending on the response
        f.write(
            server.strip() + " is alive" + "\n"
        )  # Write out that you can receive a reponse
    else:
        f.write(
            server.strip() + " did not respond" + "\n"
        )  # Write out you can't reach the box

print("\n\tYou can see the results in the logfile : " + logfilename)
# Show the location of the logfile
