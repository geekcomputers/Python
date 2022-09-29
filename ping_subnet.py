from __future__ import print_function

import os  # Load the Library Module
import subprocess  # Load the Library Module
import sys  # Load the Library Module

# Script Name		: ping_subnet.py
# Author				: Craig Richards
# Created				: 12th January 2012
# Last Modified		:
# Version				: 1.0
# Modifications		:
# Description			: After supplying the first 3 octets it will scan the final range for available addresses

filename = sys.argv[0]  # Sets a variable for the script name

if (
    "-h" in sys.argv or "--h" in sys.argv or "-help" in sys.argv or "--help" in sys.argv
):  # Help Menu if called
    print(
        """
You need to supply the first octets of the address Usage : """
        + filename
        + """ 111.111.111 """
    )
    sys.exit(0)
else:

    if (
        len(sys.argv) < 2
    ):  # If no arguments are passed then display the help and instructions on how to run the script
        sys.exit(
            f" You need to supply the first octets of the address Usage : {filename} 111.111.111"
        )


    if os.name == "posix":  # Check the os, if it's linux then
        myping = "ping -c 2 "  # This is the ping command
    elif os.name in ("nt", "dos", "ce"):  # Check the os, if it's windows then
        myping = "ping -n 2 "  # This is the ping command

    subnet = sys.argv[1]
    f = open(f"ping_{subnet}.log", "w")
    for ip in range(2, 255):  # Set the ip variable for the range of numbers
        ret = subprocess.call(
            myping + str(subnet) + "." + str(ip),
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )  # Run the command pinging the servers
        if ret == 0:  # Depending on the response
            f.write(f"{subnet}.{str(ip)} is alive" + "\n")
        else:
            f.write(f"{subnet}.{str(ip)} did not respond" + "\n")
