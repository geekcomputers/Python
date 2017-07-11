# Script Name		: work_connect.py
# Author				: Craig Richards
# Created				: 11th May 2012
# Last Modified		: 31st October 2012
# Version				: 1.1

# Modifications		: 1.1 - CR - Added some extra code, to check an argument is passed to the script first of all, then check it's a valid input

# Description			: This simple script loads everything I need to connect to work etc

import subprocess				# Load the Library Module 
import sys							# Load the Library Module 
import os							# Load the Library Module
import time						# Load the Library Module

dropbox = os.getenv("dropbox")							# Set the variable dropbox, by getting the values of the environment setting for dropbox
rdpfile = ("remote\\workpc.rdp")							# Set the variable logfile, using the arguments passed to create the logfile
conffilename=os.path.join(dropbox, rdpfile)			# Set the variable conffilename by joining confdir and conffile together
remote = (r"c:\windows\system32\mstsc.exe ")	# Set the variable remote with the path to mstsc

text = '''You need to pass an argument
	-c Followed by login password to connect
	-d to disconnect'''											# Text to display if there is no argument passed or it's an invalid option - 1.2

if len(sys.argv) < 2:											# Check there is at least one option passed to the script - 1.2
  print text															# If not print the text above - 1.2
  sys.exit()														# Exit the program - 1.2
  
if '-h' in sys.argv or '--h' in sys.argv or '-help' in sys.argv or '--help' in sys.argv:	# Help Menu if called
    print text														# Print the text, stored in the text variable - 1.2
    sys.exit(0)														# Exit the program
else:
  if sys.argv[1].lower().startswith('-c'):					# If the first argument is -c then
    passwd = sys.argv[2] 									# Set the variable passwd as the second argument passed, in this case my login password
    subprocess.Popen((r"c:\Program Files\Checkpoint\Endpoint Connect\trac.exe connect -u username -p "+passwd))
    subprocess.Popen((r"c:\geektools\puttycm.exe"))
    time.sleep(15)												# Sleep for 15 seconds, so the checkpoint software can connect before opening mstsc
    subprocess.Popen([remote, conffilename])
  elif sys.argv[1].lower().startswith('-d'):					# If the first argument is -d then disconnect my checkpoint session.
    subprocess.Popen((r"c:\Program Files\Checkpoint\Endpoint Connect\trac.exe disconnect "))
  else:
    print 'Unknown option - ' + text						# If any other option is passed, then print Unknown option and the text from above - 1.2