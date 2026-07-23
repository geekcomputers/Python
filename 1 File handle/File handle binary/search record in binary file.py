# binary file to search a given record

import pickle
from dotenv import load_dotenv


def search():
    with open("student_records.pkl", "rb") as F:
        # your file path will be different
        search = True
        rno = int(input("Enter the roll number of the student"))

        for i in pickle.load(F):
            if i[0] == rno:
                print(f"Record found successfully\n{i}")
                search = False

        if search:
            print("Sorry! record not found")


binary_search()
