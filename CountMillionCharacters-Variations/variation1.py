try:
    input = raw_input
except NameError:
    pass


def count_chars(filename):
    count = {}

    with open(filename) as info:  # inputFile Replaced with filename
        readfile = info.read()
        for character in readfile.upper():
            count[character] = count.get(character, 0) + 1

    return count


def main():
    is_exist = True
    # Try to open file if exist else raise exception and try again
    while (is_exist):
        try:
            inputFile = input("File Name / (0)exit : ").strip()
            if inputFile == "0":
                break
            print(count_chars(inputFile))
        except FileNotFoundError:
            print("File not found...Try again!")


if __name__ == '__main__':
    main()
