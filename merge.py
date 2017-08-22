# author:zhangshuyx@gmail.com

#!/usr/bin/env python
# -*- coding=utf-8 -*-

import os

# define the result filename
resultfile = 'result.csv'

# the merge func
def merge():
    """merge csv files to one file"""
    # use list save the csv files
    csvfiles = [f for f in os.listdir('.') if f != resultfile and f.split('.')[1]=='csv']
    # open file to write
    with open(resultfile,'w') as writefile:
        for csvfile in csvfiles:
            with open(csvfile) as readfile:
                print('File {} readed.'.format(csvfile))
                # do the read and write
                writefile.write(readfile.read()+'\n')
    print('\nFile {} wrote.'.format(resultfile))

# the main program
if __name__ == '__main__':
    merge()
