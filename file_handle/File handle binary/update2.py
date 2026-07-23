import pickle
import os
from dotenv import load_dotenv

base = os.path.dirname(__file__)
load_dotenv(os.path.join(base, ".env"))
student_record = os.getenv("STUDENTS_RECORD_FILE")


def update():
    with open(student_record, "rb") as F:
        S = pickle.load(F)
        found = False
        rno = int(input("enter the roll number you want to update"))

        for i in S:
            if rno == i[0]:
                print(f"the currrent name is {i[1]}")
                i[1] = input("enter the new name")
                found = True
                break

        if found:
            print("Record not found")

        with open(student_record, "wb") as F:
            pickle.dump(S, F)


update()
