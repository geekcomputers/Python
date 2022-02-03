"""
Author:         Linjian Li (github.com/LinjianLi)
Created:        2020-04-09
Last Modified:  2020-10-17
Description:    A script that replace specified text string in a text file.
How to use:     `-f` specifying the text file,
                `-e` specifying the encoding (optional),
                `-o` specifying the old text string to be replaced),
                `-n` specifying the new text string to replace with.
"""


def text_file_replace(file, encoding, old, new):
    lines = []
    cnt = 0
    with open(file=file, mode="r", encoding=encoding) as fd:
        for line in fd:
            cnt += line.count(old)
            lines.append(line.replace(old, new))
    with open(file=file, mode="w", encoding=encoding) as fd:
        fd.writelines(lines)
    print('{} occurence(s) of "{}" have been replaced with "{}"'.format(cnt, old, new))
    return cnt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File.")
    parser.add_argument("-e", "--encoding", default="utf-8", help="Encoding.")
    parser.add_argument("-o", "--old", help="Old string.")
    parser.add_argument("-n", "--new", help="New string.")
    args = parser.parse_args()

    text_file_replace(args.file, args.encoding, args.old, args.new)
