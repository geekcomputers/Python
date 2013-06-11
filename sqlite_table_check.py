# Script Name	: sqlite_table_check.py
# Author		: Craig Richards
# Created		: 07 June 2013
# Last Modified	:
# Version		: 1.0

# Modifications	:

# Description	: Checks the main SQLITE database to ensure all the tables should exist


import sqlite3 
import sys
import os

dropbox= os.getenv("dropbox")
config=os.getenv("my_config")
dbfile=("Databases\jarvis.db")
listfile=("sqlite_master_table.lst")
master_db=os.path.join(dropbox, dbfile)
config_file=os.path.join(config, listfile)
tablelist=open(config_file,'r');

conn = sqlite3.connect(master_db)
cursor = conn.cursor()
cursor.execute('SELECT SQLITE_VERSION()')
data = cursor.fetchone()

if str(data) == "(u'3.6.21',)":
  print ("\nCurrently " + master_db + " is on SQLite version: %s" % data + " - OK -\n")
else:
  print ("\nDB On different version than master version - !!!!! \n")
conn.close()

print ("\nCheckling " + master_db + " against " + config_file + "\n")

for table in tablelist.readlines():
  conn = sqlite3.connect(master_db)
  cursor = conn.cursor()
  cursor.execute("select count(*) from sqlite_master where name = ?",(table.strip(), ))
  res = cursor.fetchone()
    
  if (res[0]):
    print ('[+] Table : ' + table.strip() + ' exists [+]')
  else:
    print ('[-] Table : ' + table.strip() + '  does not exist [-]')
  