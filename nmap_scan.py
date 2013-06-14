# Script Name		: nmap_scan.py
# Author				: Craig Richards
# Created				: 24th May 2013
# Last Modified		: 
# Version				: 1.0

# Modifications		: 

# Description			: This scans my scripts directory and gives a count of the different types of scripts, you need nmap installed to run this

import nmap			# Import the module
import optparse		# Import the module

def nmapScan(tgtHost, tgtPort):									# Create the function, this fucntion does the scanning
  nmScan = nmap.PortScanner()
  nmScan.scan(tgtHost, tgtPort)
  state = nmScan[tgtHost]['tcp'][int(tgtPort)]['state']
  print "[*] " + tgtHost + " tcp/"+tgtPort +" "+state
  
def main():																								# Main Program
  parser = optparse.OptionParser('usage%prog ' + '-H <host> -p <port>')		# Display options/help if required
  parser.add_option('-H', dest='tgtHost', type='string', help='specify host')
  parser.add_option('-p', dest='tgtPort', type='string', help='port')
  (options, args) = parser.parse_args()
  tgtHost = options.tgtHost
  tgtPorts = str(options.tgtPort).split(',')
  
  if (tgtHost == None) | (tgtPorts[0] == None):  
    print parser.usage
    exit(0)

  for tgtPort in tgtPorts:																			# Scan the hosts with the ports etc
    nmapScan(tgtHost, tgtPort)

if __name__ == '__main__':
  main()