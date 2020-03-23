# Script Name       : fileinfo.py
# Author            : Not sure where I got this from
# Created           : 28th November 2011
# Last Modified     : 23th March 2020
# Version           : 1.0
# Modifications     :

# Description       : Show file information for a given file

# get file information using os.stat()
# tested with Python24 vegsaeat 25sep2006
from __future__ import print_function

import os
import stat  # index constants for os.stat()
import sys
import time
from collections import OrderedDict

if sys.version_info >= (3, 0):
    raw_input = input

# pick a file you have
file_name = raw_input("Enter a file name: ")  
count = 0
t_char = 0

try:
    with open(file_name) as f:
        # Source: https://stackoverflow.com/a/1019572
        count = (sum(1 for line in f))
        f.seek(0)
        t_char = (sum([len(line) for line in f]))
except FileNotFoundError as e:
    print(e)
    sys.exit(1)
# When open item is a directory (python2)
except IOError:
    pass
# When open item is a directory (python3)
except IsADirectoryError:
    pass

file_stats = os.stat(file_name)
# create a dictionary to hold file info
file_info = {
    'fname': file_name,
    'fsize': file_stats[stat.ST_SIZE],
    'f_lm': time.strftime("%d/%m/%Y %I:%M:%S %p",
                          time.localtime(file_stats[stat.ST_MTIME])),
    'f_la': time.strftime("%d/%m/%Y %I:%M:%S %p",
                          time.localtime(file_stats[stat.ST_ATIME])),
    'f_ct': time.strftime("%d/%m/%Y %I:%M:%S %p",
                          time.localtime(file_stats[stat.ST_CTIME])),
    'no_of_lines': count,
    't_char': t_char
}

file_info_fmt = ('\nfile name = {}\nfile size = {}\nlast modified = {}\n'
                 'last accessed = {}\ncreation time = {}\n'
                 'Total number of lines are = {}\n'
                 'Total number of characters are = {}')
print(
    file_info_fmt.format(file_info['fname'],
                         str(file_info['fsize']) + " bytes", file_info['f_lm'],
                         file_info['f_la'], file_info['f_ct'],
                         file_info['no_of_lines'], file_info['t_char']))

if stat.S_ISDIR(file_stats[stat.ST_MODE]):
    print("This a directory")
else:
    file_stats_fmt = ''
    print("\nThis is not a directory.")
    stats_strings = ("st_mode (protection bits),"
                     "st_ino (inode number),"
                     "st_dev (device),"
                     "st_nlink (number of hard links),"
                     "st_uid (user ID of owner),"
                     "st_gid (group ID of owner),"
                     "st_size (file size bytes),"
                     "st_atime (last access time seconds since epoch),"
                     "st_mtime (last modification time),"
                     "st_ctime (time of creation Windows)")
    stats_keys = stats_strings.split(',')
    for key, value in OrderedDict(zip(stats_keys, file_stats)).items():
        print(key, ' =', value)
