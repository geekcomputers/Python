import sys

try:
    input_func = raw_input
except:
    input_func = input


def count_chars(filename):
    count = {}

    with open(filename) as info:  # inputFile Replaced with filename
        readfile = info.read()
        for character in readfile.upper():
            count[character] = count.get(character, 0) + 1

    return count

def main():
    is_exist=True
    #Try to open file if exist else raise exception and try again
    while(is_exist):
        try:
            inputFile = input_func("File Name / (0)exit : ")
            if inputFile == "0":
                break
            print(count_chars(inputFile))
        except FileNotFoundError:
            print("File not found...Try again!")


if __name__ == '__main__':
    main()
