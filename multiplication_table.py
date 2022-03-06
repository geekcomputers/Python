"""
The 'multiplication table' Implemented in Python 3

Syntax:
python3 multiplication_table.py [rows columns]
Separate filenames with spaces as usual.

Updated by Borys Baczewski (BB_arbuz on GitHub) - 06/03/2022
"""

from sys import argv  # import argument variable

(
    script,
    rows,
    columns,
) = argv  # define rows and columns for the table and assign them to the argument variable


def table(rows, columns):
    columns = int(columns)
    rows = int(rows)
    for r in range(1, rows+1):
        c = r
        print(r, end='\t')
        i = 0
        while columns-1 > i:
            print(c+r, end='\t')
            c = c+r
            i += 1
        print('\n')


table(rows, columns)
