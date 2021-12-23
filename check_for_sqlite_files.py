# Script Name	: check_for_sqlite_files.py
# Author		: Craig Richards
# Created		: 07 June 2013
# Last Modified	: 14 February 2016
# Version		: 1.0.1

# Modifications	: 1.0.1 - Remove unecessary line and variable on Line 21

# Description	: Scans directories to check if there are any sqlite files in there 

from __future__ import print_function

import os


def isSQLite3(filename):
    """
    Checks if the given file is a SQLite database.

    :param filename: The name of the file to be checked.
    :type filename: str

        :returns bool -- True if
    it's a SQLite database, False otherwise.
    """
    from os.path import isfile, getsize

    if not isfile(filename):
        return False
    if getsize(filename) < 100:  # SQLite database file header is 100 bytes
        return False
    else:
        fd = open(filename, 'rb')
        header = fd.read(100)
        fd.close()

        if header[0:16] == 'SQLite format 3\000':
            return True
        else:
            return False


log = open('sqlite_audit.txt', 'w')
for r, d, f in os.walk(r'.'):
    for files in f:
        if isSQLite3(files):
            print(files)
            print("[+] '%s' **** is a SQLITE database file **** " % os.path.join(r, files))
            log.write("[+] '%s' **** is a SQLITE database file **** " % files + '\n')
        else:
            log.write("[-] '%s' is NOT a sqlite database file" % os.path.join(r, files) + '\n')
            log.write("[-] '%s' is NOT a sqlite database file" % files + '\n')
