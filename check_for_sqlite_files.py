# Script Name	: check_for_sqlite_files.py
# Author		: Craig Richards
# Created		: 07 June 2013
# Last Modified	: 14 February 2016
# Version		: 1.0.1

# Modifications	: 1.0.1 - Remove unecessary line and variable on Line 21

# Description	: Scans directories to check if there are any sqlite files in there 

from __future__ import print_function
from os.path import isfile, getsize

import os

def isSQLite3(filename):
    # SQLite database file header is 100 bytes
    if not isfile(filename) or getsize(filename) < 100:
        return False
    else:
        with open(filename, 'rb') as fd:
            Header = fd.read(100)
            return Header[0:16] == 'SQLite format 3\000'

log=open('sqlite_audit.txt','w')
for path, dirs, files in os.walk(r'.'):
  for f in files:
    if isSQLite3(f):
      print(f)
      print("[+] '%s' **** is a SQLITE database file **** " % os.path.join(path, f))
      log.write("[+] '%s' **** is a SQLITE database file **** " % f + '\n')
    else:
      log.write("[-] '%s' is NOT a sqlite database file" % os.path.join(path, f) + '\n')
      log.write("[-] '%s' is NOT a sqlite database file" % f + '\n')