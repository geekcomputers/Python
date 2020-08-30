# batch_file_rename.py
# Created: 6th August 2012

"""
This will batch rename a group of files in a given directory,
once you pass the current and new extensions
"""

# just checking
__author__ = 'Craig Richards'
__version__ = '1.0'

import argparse
import os


def batch_rename(work_dir, old_ext, new_ext):
    """
    This will batch rename a group of files in a given directory,
    once you pass the current and new extensions
    """
    # files = os.listdir(work_dir)
    for filename in os.listdir(work_dir):
        # Get the file extension
        split_file = os.path.splitext(filename)
        # Unpack tuple element
        root_name, file_ext = split_file
        # Start of the logic to check the file extensions, if old_ext = file_ext
        if old_ext == file_ext:
            # Returns changed name of the file with new extention
            newfile = root_name + new_ext

            # Write the files
            os.rename(
                os.path.join(work_dir, filename),
                os.path.join(work_dir, newfile)
            )
    print("rename is done!")
    print(os.listdir(work_dir))


def get_parser():
    parser = argparse.ArgumentParser(description='change extension of files in a working directory')
    parser.add_argument('work_dir', metavar='WORK_DIR', type=str, nargs=1,
                        help='the directory where to change extension')
    parser.add_argument('old_ext', metavar='OLD_EXT', type=str, nargs=1, help='old extension')
    parser.add_argument('new_ext', metavar='NEW_EXT', type=str, nargs=1, help='new extension')
    return parser


def main():
    """
    This will be called if the script is directly invoked.
    """
    # adding command line argument
    parser = get_parser()
    args = vars(parser.parse_args())

    # Set the variable work_dir with the first argument passed
    work_dir = args['work_dir'][0]
    # Set the variable old_ext with the second argument passed
    old_ext = args['old_ext'][0]
    if old_ext and old_ext[0] != '.':
        old_ext = '.' + old_ext
    # Set the variable new_ext with the third argument passed
    new_ext = args['new_ext'][0]
    if new_ext and new_ext[0] != '.':
        new_ext = '.' + new_ext

    batch_rename(work_dir, old_ext, new_ext)


if __name__ == '__main__':
    main()
