# updating records in a binary file

import pickle
import os

base = os.path.dirname(__file__)
from dotenv import load_dotenv

load_dotenv(os.path.join(base, ".env"))
student_record = os.getenv("STUDENTS_RECORD_FILE")

## ! Understand how pandas works internally


def update():
    with open(student_record, "rb") as File:
        value = pickle.load(File)
        found = False
        roll = int(input("Enter the roll number of the record"))

        for i in value:
            if roll == i[0]:
                print(f"current name {i[1]}")
                print(f"current marks {i[2]}")
                i[1] = input("Enter the new name")
                i[2] = int(input("Enter the new marks"))
                found = True

        if not found:
            print("Record not found")

    with open(student_record, "wb") as File:
        pickle.dump(value, File)


update()

# ! Instead of AB use WB?
# ! It may have memory limits while updating large files but it would be good
# ! Few lakhs records would be fine and wouldn't create any much of a significant issues
