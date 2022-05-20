""" author: Ataba29 
code has a matrix each list inside of the matrix has two strings
the code determines if the two strings are similar or different 
from each other recursively
"""


def CheckTwoStrings(str1, str2):
    # function takes two strings and check if they are similar
    # returns True if they are identical and False if they are different

    if(len(str1) != len(str2)):
        return False
    if(len(str1) == 1 and len(str2) == 1):
        return str1[0] == str2[0]

    return (str1[0] == str2[0]) and CheckTwoStrings(str1[1:], str2[1:])


def main():
    matrix = [["hello", "wow"], ["ABSD", "ABCD"],
              ["List", "List"], ["abcspq", "zbcspq"],
              ["1263", "1236"], ["lamar", "lamars"],
              ["amczs", "amczs"], ["yeet", "sheesh"], ]

    for i in matrix:
        if CheckTwoStrings(i[0], i[1]):
            print(f"{i[0]},{i[1]} are similar")
        else:
            print(f"{i[0]},{i[1]} are different")


main()
