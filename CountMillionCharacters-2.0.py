"""Get the number of each character in any given text.
Inputs:
A txt file -- You will be asked for an input file. Simply input the name
of the txt file in which you have the desired text.
"""

import collections
import pprint


def main():
    """
    This function takes a file name as an input and counts the number of times each letter appears in the text.
    """
    file_input = input('File Name: ')
    try:
        with open(file_input, 'r') as info:
            count = collections.Counter(info.read().upper())
    except FileNotFoundError:
        print("Please enter a valid file name.")
        main()

    value = pprint.pformat(count)
    print(value)
    exit()


if __name__ == "__main__":
    main()
