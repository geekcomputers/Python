"""Rangoli Model"""


# Prints a rangoli of size n
def print_rangoli(n):
    """Prints a rangoli of size n"""    
    # Width of the rangoli
    width = 4 * n - 3

    # String to be printed
    string = ""

    # Loop to print the rangoli
    for i in range(1, n + 1):
        for j in range(0, i):
            string += chr(96 + n - j)
            if len(string) < width:
                string += "-"

        for k in range(i - 1, 0, -1):
            string += chr(97 + n - k)
            if len(string) < width:
                string += "-"

        print(string.center(width, "-"))
        string = ""

    for i in range(n - 1, 0, -1):
        for j in range(0, i):
            string += chr(96 + n - j)
            if len(string) < width:
                string += "-"

        for k in range(i - 1, 0, -1):
            string += chr(97 + n - k)
            if len(string) < width:
                string += "-"

        print(string.center(width, "-"))
        string = ""


if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)
