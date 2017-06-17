import sys


def countchars(filename):
    count = {}

    with open(filename) as info:  # inputFile Replaced with filename
        readfile = info.read()
        for character in readfile.upper():
            count[character] = count.get(character, 0) + 1

    return count

if __name__ == '__main__':
    if sys.version_info.major >= 3:  # if the interpreter version is 3.X, use 'input',
        input_func = input           # otherwise use 'raw_input'
    else:
        input_func = raw_input

    inputFile = input_func("File Name : ")
    print(countchars(inputFile))
