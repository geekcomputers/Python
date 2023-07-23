# binary file to search a given record

import pickle


def binary_search():
    F = open("studrec.dat", "rb")
    # your file path will be different
    value = pickle.load(F)
    search = 0
    rno = int(input("Enter the roll number of the student"))

    for i in value:
        if i[0] == rno:
            print("Record found successfully")
            print(i)
            search = 1

    if search == 0:
        print("Sorry! record not found")
    F.close()


binary_search()
