"""Write a function in python to count the number of lowercase
alphabets present in a text file “happy.txt"""


def lowercase():
    with open("happy.txt") as F:
        count_lower = 0
        count_upper = 0
        value = F.read()
        for i in value:
            if i.islower():
                count_lower += 1
            elif i.isupper():
                count_upper += 1
        print("The total number of lower case letters are", count_lower)
        print("The total number of upper case letters are", count_upper)
        print("The total number of letters are", count_lower + count_upper)

if __name__ == "__main__":
    lowercase()
