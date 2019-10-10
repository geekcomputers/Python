from __future__ import print_function

import optparse  # Import the module
from socket import *  # Import the module
from threading import *  # Import the module

# Script Name	: portscanner.py
# Author		: Craig Richards
# Created		: 20 May 2013
# Last Modified	:
# Version		: 1.0
# Modifications	:
# Description	: Port Scanner, you just pass the host and the ports

screenLock = Semaphore(value=1)  # Prevent other threads from preceeding


def connScan(tgtHost, tgtPort):  # Start of the function
    try:
        connSkt = socket(AF_INET, SOCK_STREAM)  # Open a socket
        connSkt.connect((tgtHost, tgtPort))
        connSkt.send('')
        results = connSkt.recv(100)
        screenLock.acquire()  # Acquire the lock
        print('[+] %d/tcp open' % tgtPort)
        print('[+] ' + str(results))
    except:
        screenLock.acquire()
        print('[-] %d/tcp closed ' % tgtPort)
    finally:
        screenLock.release()
        connSkt.close()


def portScan(tgtHost, tgtPorts):  # Start of the function
    try:
        tgtIP = gethostbyname(tgtHost)  # Get the IP from the hostname
    except:
        print("[-] Cannot resolve '%s': Unknown host" % tgtHost)
        return
    try:
        tgtName = gethostbyaddr(tgtIP)  # Get hostname from IP
        print('\n[+] Scan Results for: ' + tgtName[0])
    except:
        print('\n[+] Scan Results for: ' + tgtIP)
    setdefaulttimeout(1)
    for tgtPort in tgtPorts:  # Scan host and ports
        t = Thread(target=connScan, args=(tgtHost, int(tgtPort)))
        t.start()


def main():
    parser = optparse.OptionParser('usage %prog -H' + ' <target host> -p <target port>')
    parser.add_option('-H', dest='tgtHost', type='string', help='specify target host')
    parser.add_option('-p', dest='tgtPort', type='string', help='specify target port[s] seperated by a comma')
    (options, args) = parser.parse_args()
    tgtHost = options.tgtHost
    tgtPorts = str(options.tgtPort).split(',')
    if (tgtHost == None) | (tgtPorts[0] == None):
        print(parser.usage)
        exit(0)
    portScan(tgtHost, tgtPorts)


if __name__ == '__main__':
    main()
