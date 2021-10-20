from __future__ import print_function

import os

# author:zhangshuyx@gmail.com
# !/usr/bin/env python
# -*- coding=utf-8 -*-

# define the result filename
resultfile = 'result.csv'


# the merge func
def merge():
    """merge csv files to one file"""

    # indicates use of a global variable.
    global resultfile

    # use list save the csv files
    csvfiles = [f for f in os.listdir('.') if f != resultfile \
                and (len(f.split('.')) >= 2) and f.split('.')[1] == 'csv']

    # open file to write
    with open(resultfile, 'w') as writefile:
        for csvfile in csvfiles:
            with open(csvfile) as readfile:
                print('File {} readed.'.format(csvfile))

                # do the read and write
                writefile.write(readfile.read() + '\n')
    print('\nFile {} wrote.'.format(resultfile))


# the main program

def main():
    print("\t\tMerge\n\n")
    print("This program merges csv-files to one file\n")
    merge()


if __name__ == '__main__':
    main()
