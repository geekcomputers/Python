# Script Name	: sqlite_check.py
# Author		: Craig Richards
# Created		: 20 May 2013
# Last Modified	:
# Version		: 1.0

# Modifications	:

# Description	: Runs checks to check my SQLITE database


import sqlite3 as lite
import sys
import os

dropbox= os.getenv("dropbox")
dbfile=("Databases\jarvis.db")
master_db=os.path.join(dropbox, dbfile)
con = None

try:
    con = lite.connect(master_db)
    cur = con.cursor()
    cur.execute('SELECT SQLITE_VERSION()')
    data = cur.fetchone()
    print "SQLite version: %s" % data


except lite.Error, e:

    print "Error %s:" % e.args[0]
    sys.exit(1)

finally:

    if con:
        con.close()


con = lite.connect(master_db)
cur=con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
rows = cur.fetchall()
for row in rows:
  print row

con = lite.connect(master_db)
cur=con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
while True:
  row = cur.fetchone()
  if row == None:
    break
  print row[0]