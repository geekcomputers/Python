# Script Name		: testlines.py
# Author		: Craig Richards
# Created		: 08th December 2011
# Last Modified		: 
# Version		: 1.1

# Modifications		: beven nyamande
#                  : Scott

# Description		: This is a very simple script that opens up a file and writes whatever is set "
import sys

def write_to_file(filename, reps, txt):
    with open(filename, 'w') as file_object:
        for _ in range(reps):
            s = file_object.write(txt if txt.endswith('\n') else txt + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 testlines.py file lines line")
        exit()
    if not sys.argv[2].isdigit():
        print(f"{sys.argv[2]} is not an integer, please correct")
        exit()
    write_to_file(sys.argv[1], int(sys.argv[2]), ' '.join(sys.argv[1::len(sys.argv)]))
