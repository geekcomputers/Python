# Script Name		: fileinfo.py
# Author				: Not sure where I got this from
# Created				: 28th November 2011
# Last Modified		: 
# Version				: 1.0
# Modifications		: 

# Description			: Show file information for a given file


# get file information using os.stat()
# tested with Python24 vegsaeat 25sep2006
import os
import stat # index constants for os.stat()
import time
# pick a file you have ...
file_name = raw_input("Enter a file name: ")
file_stats = os.stat(file_name)
# create a dictionary to hold file info
file_info = {
  'fname': file_name,
  'fsize': file_stats [stat.ST_SIZE],
  'f_lm': time.strftime("%d/%m/%Y %I:%M:%S %p",time.localtime(file_stats[stat.ST_MTIME])),
  'f_la': time.strftime("%d/%m/%Y %I:%M:%S %p",time.localtime(file_stats[stat.ST_ATIME])),
  'f_ct': time.strftime("%d/%m/%Y %I:%M:%S %p",time.localtime(file_stats[stat.ST_CTIME]))
}
print
print "file name = %(fname)s" % file_info
print "file size = %(fsize)s bytes" % file_info
print "last modified = %(f_lm)s" % file_info
print "last accessed = %(f_la)s" % file_info
print "creation time = %(f_ct)s" % file_info
print
if stat.S_ISDIR(file_stats[stat.ST_MODE]):
  print "This a directory"
else:
  print "This is not a directory"
  print
  print "A closer look at the os.stat(%s) tuple:" % file_name
  print file_stats
  print
  print "The above tuple has the following sequence:"
  print """st_mode (protection bits), st_ino (inode number),
  st_dev (device), st_nlink (number of hard links),
  st_uid (user ID of owner), st_gid (group ID of owner),
  st_size (file size, bytes), st_atime (last access time, seconds since epoch),
  st_mtime (last modification time), st_ctime (time of creation, Windows)"""
