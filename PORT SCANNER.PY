PORT SCANNER IN PYTHON...
This post will show how you can make a small and easy-to-use port scanner program
written in Python.
There are many ways of doing this with Python, and I'm going to do it using the
built-in module Socket.
Sockets
The socket module in Python provides access to the BSD socket interface. 

It includes the socket class, for handling the actual data channel, and functions 
for network-related tasks such as converting a server’s name to an address and
formatting data to be sent across the network. Source

Sockets are widely used on the Internet, as they are behind any kind of
network communications done by your computer. 

The INET sockets, account for at least 99% of the sockets in use. 

The web browser’s that you use opens a socket and connects to the web server.

Any network communication goes through a socket.

For more reading about the socket module, please see the official documentation
Socket functions
Before we get started with our sample program, let's see some of the socket
functions we are going to use.

sock = socket.socket (socket_family, socket_type)
Syntax for creating a socket

sock = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
Creates a stream socket

AF_INET
Socket Family (here Address Family version 4 or IPv4)

SOCK_STREAM
Socket type TCP connections

SOCK_DGRAM
Socket type UDP connections

gethostbyname("host")
Translate a host name to IPv4 address format

socket.gethostbyname_ex("host")
Translate a host name to IPv4 address format, extended interface

socket.getfqdn("8.8.8.8")
Get the fqdn (fully qualified domain name)

socket.gethostname()
Returns the hostname of the machine..

socket.error
Exception handling
Making a program using Python Sockets
How to make a simple port scanner program in Python

This small port scanner program will try to connect on every port you define for
a particular host. 

The first thing we must do is import the socket library and other libraries that
we need. 

Open up an text editor, copy & paste the code below. Save the file as:
"portscanner.py" and exit the editor
#!/usr/bin/env python
import socket
import subprocess
import sys
from datetime import datetime

# Clear the screen
subprocess.call('clear', shell=True)

# Ask for input
remoteServer    = raw_input("Enter a remote host to scan: ")
remoteServerIP  = socket.gethostbyname(remoteServer)

# Print a nice banner with information on which host we are about to scan
print "-" * 60
print "Please wait, scanning remote host", remoteServerIP
print "-" * 60

# Check what time the scan started
t1 = datetime.now()

# Using the range function to specify ports (here it will scans all ports between 1 and 1024)

# We also put in some error handling for catching errors

try:
    for port in range(1,1025):  
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((remoteServerIP, port))
        if result == 0:
            print "Port {}: 	 Open".format(port)
        sock.close()

except KeyboardInterrupt:
    print "You pressed Ctrl+C"
    sys.exit()

except socket.gaierror:
    print 'Hostname could not be resolved. Exiting'
    sys.exit()

except socket.error:
    print "Couldn't connect to server"
    sys.exit()

# Checking the time again
t2 = datetime.now()

# Calculates the difference of time, to see how long it took to run the script
total =  t2 - t1

# Printing the information to screen
print 'Scanning Completed in: ', total

