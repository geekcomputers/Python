# batch_file_rename.py
# Created: 6th August 2012
#Modified: 27th January 2016 by Gregory Dolan

'''
This will batch rename a group of files in a given directory,
once you pass the current and new extensions
'''

__author__ = 'Craig Richards'
__version__ = '1.0'

import os
import sys


def batch_rename(work_dir, old_ext, new_ext):
    '''
    This will batch rename a group of files in a given directory,
    once you pass the current and new extensions
    '''
    for filename in os.listdir(work_dir):
        # Get the file extension
        file_ext = os.path.splitext(filename)[1]
        # Start of the logic to check the file extensions, if old_ext = file_ext
        if old_ext == file_ext:
            # Set newfile to be the filename, replaced with the new extension
            newfile = filename.replace(old_ext, new_ext)
            # Write the files
            os.rename(
              os.path.join(work_dir, filename),
              os.path.join(work_dir, newfile)
            )


def main():
    '''
    This will be called if the script is directly invoked.
    '''
    if len(sys.argv) == 4:
        # Set the variable work_dir with the first argument passed
        work_dir = sys.argv[1]
        # Set the variable old_ext with the second argument passed
        old_ext = sys.argv[2]
        # Set the variable new_ext with the third argument passed
        new_ext = sys.argv[3]
        batch_rename(work_dir, old_ext, new_ext)
    else:
        print 'Usage:', str(sys.argv[0]), '<work_dir> <old_ext> <new_ext>'
        exit(0)

if __name__ == '__main__':
    main()
