"""Get the number of each character in any given text.

Inputs:
A txt file -- You will be asked for an input file. Simply input the name
of the txt file in which you have the desired text.

"""

import pprint
import collections


def main():

    file_input = input('File Name: ')

    with open(file_input, 'r') as info:
        count = collections.Counter(info.read().upper())

    value = pprint.pformat(count)
    print(value)


if __name__ == "__main__":
    main()
