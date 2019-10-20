# Script Name   : move_files_over_x_days.py
# Author(s)     : Craig Richards ,Demetrios Bairaktaris
# Created       : 8th December 2011
# Last Modified : 25 December 2017
# Version       : 1.1
# Modifications : Added possibility to use command line arguments to specify source, destination, and days. 
# Description   : This will move all the files from the src directory that are over 240 days old to the destination directory.

import argparse
import os
import shutil
import time

usage = 'python move_files_over_x_days.py -src [SRC] -dst [DST] -days [DAYS]'
description = 'Move files from src to dst if they are older than a certain number of days.  Default is 240 days'

args_parser = argparse.ArgumentParser(usage=usage, description=description)
args_parser.add_argument('-src', '--src', type=str, nargs='?', default='.',
                         help='(OPTIONAL) Directory where files will be moved from. Defaults to current directory')
args_parser.add_argument('-dst', '--dst', type=str, nargs='?', required=True,
                         help='(REQUIRED) Directory where files will be moved to.')
args_parser.add_argument('-days', '--days', type=int, nargs='?', default=240,
                         help='(OPTIONAL) Days value specifies the minimum age of files to be moved. Default is 240.')
args = args_parser.parse_args()

if args.days < 0:
    args.days = 0

src = args.src  # Set the source directory
dst = args.dst  # Set the destination directory
days = args.days  # Set the number of days
now = time.time()  # Get the current time

if not os.path.exists(dst):
    os.mkdir(dst)

for f in os.listdir(src):  # Loop through all the files in the source directory
    if os.stat(f).st_mtime < now - days * 86400:  # Work out how old they are, if they are older than 240 days old
        if os.path.isfile(f):  # Check it's a file
            shutil.move(f, dst)  # Move the files
