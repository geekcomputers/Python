# Script Name	: python_sms.py
# Author	: Craig Richards
# Created	: 16th February 2017
# Last Modified	: 
# Version	: 1.0

# Modifications	: 

# Description	: This will text all the students Karate Club

import urllib      # URL functions
import urllib2     # URL functions
import os
from time import strftime
import sqlite3
import sys

dropbox = os.getenv("dropbox")
scripts = os.getenv("scripts")
dbfile = ("database/maindatabase.db")
master_db = os.path.join(dropbox, dbfile)

f = open(scripts+'/output/student.txt','a')

tdate = strftime("%d-%m")

conn = sqlite3.connect(master_db)
cursor = conn.cursor()
loc_stmt = 'SELECT name, number from table'
cursor.execute(loc_stmt)
while True:							
  row = cursor.fetchone()	
  if row == None:
    break
  sname = row[0]
  snumber = row[1]

  message = (sname + ' There will be NO training tonight on the ' + tdate + ' Sorry for the late notice, I have sent a mail as well, just trying to reach everyone, please do not reply to this message as this is automated')

  username = 'YOUR_USERNAME'
  sender = 'WHO_IS_SENDING_THE_MAIL'

  hash = 'YOUR HASH YOU GET FROM YOUR ACCOUNT'

  numbers = (snumber)

# Set flag to 1 to simulate sending, this saves your credits while you are testing your code. # To send real message set this flag to 0
  test_flag = 0

#-----------------------------------
# No need to edit anything below this line
#-----------------------------------

  values = {'test'    : test_flag,
          'uname'   : username,
          'hash'    : hash,
          'message' : message,
          'from'    : sender,
          'selectednums' : numbers }

  url = 'http://www.txtlocal.com/sendsmspost.php'

  postdata = urllib.urlencode(values)
  req = urllib2.Request(url, postdata)

  print ('Attempting to send SMS to '+ sname + ' at ' + snumber + ' on ' + tdate)
  f.write ('Attempting to send SMS to '+ sname + ' at ' + snumber + ' on ' + tdate + '\n')

  try:
    response = urllib2.urlopen(req)
    response_url = response.geturl()
    if response_url == url:
      print 'SMS sent!'
  except urllib2.URLError, e:
    print 'Send failed!'
    print e.reason
