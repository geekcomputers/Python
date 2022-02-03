# Script Name   : random_file_move.py
# Author(s)     : Akash Jain
# Created       : 1 September 2020
# Last Modified : 1 September 2020
# Version       : 1.0
# Description   : This will move specified number of files(given in ratio) from the src directory to dest directory.


import os, random
import argparse


def check_ratio(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x))
    return x


desc = "Script to move specified number of files(given in ratio) from the src directory to dest directory."
usage = "python random_file_move.py -src [SRC] -dest [DEST] -ratio [RATIO]"

parser = argparse.ArgumentParser(usage=usage, description=desc)
parser.add_argument(
    "-src",
    "--src",
    type=str,
    required=True,
    help="(REQUIRED) Path to directory from which we cut files. Space not allowed in path.",
)
parser.add_argument(
    "-dest",
    "--dest",
    type=str,
    required=True,
    help="(REQUIRED) Path to directory to which we move files. Space not allowed in path.",
)
parser.add_argument(
    "-ratio",
    "--ratio",
    type=check_ratio,
    required=True,
    help="(REQUIRED) Ratio of files in 'src' and 'dest' directory.",
)

args = parser.parse_args()

src = args.src
dest = args.dest
ratio = args.ratio

files = os.listdir(src)
size = int(ratio * len(files))

print("Move {} files from {} to {} ? [y/n]".format(size, src, dest))
if input().lower() == "y":
    for f in random.sample(files, size):
        try:
            os.rename(os.path.join(src, f), os.path.join(dest, f))
        except Exception as e:
            print(e)
    print("Successful")
else:
    print("Cancelled")
