# binary file to search a given record

import pickle


def binary_search():
    with open("studrec.dat", "rb") as F:
        # your file path will be different
        search = 0
        rno = int(input("Enter the roll number of the student"))

        for i in pickle.load(F):
            if i[0] == rno:
                print(f"Record found successfully\n{i}")
                search = 1

        if search == 0:
            print("Sorry! record not found")


binary_search()
