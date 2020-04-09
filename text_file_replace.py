"""
Author:         Linjian Li (github.com/LinjianLi)
Created:        2020-04-09
Last Modified:  2020-04-09

Description:    A script that replace specified text string in a text file.

How to use:     `-f` specifying the text file,
                `-e` specifying the encoding (optional),
                `-o` specifying the old text string to be replaced),
                `-n` specifying the new text string to replace with.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help = "File.")
parser.add_argument("-e", "--encoding", default='utf-8', help = "Encoding.")
parser.add_argument("-o", "--old", help = "Old string.")
parser.add_argument("-n", "--new", help = "New string.")
args = parser.parse_args()

f = args.file
e = args.encoding
o = args.old
n = args.new

lines = []
with open(file=f, mode='r', encoding=e) as fd:
    for line in fd:
        lines.append(line.replace(o, n))

with open(file=f, mode='w', encoding=e) as fd:
    fd.writelines(lines)