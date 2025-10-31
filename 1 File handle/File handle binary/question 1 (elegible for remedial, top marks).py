"""Amit is a monitor of class XII-A and he stored the record of all
the students of his class in a file named “student_records.pkl”.
Structure of record is [roll number, name, percentage]. His computer
teacher has assigned the following duty to Amit

Write a function remcount( ) to count the number of students who need
 remedial class (student who scored less than 40 percent)
and find the top students of the class.

We have to find weak students and bright students.
"""

## Find bright students and weak students

from dotenv import load_dotenv
import os

base = os.path.dirname(__file__)
load_dotenv(os.path.join(base, ".env"))
student_record = os.getenv("STUDENTS_RECORD_FILE")

import pickle
import logging

# Define logger with info
# import polar


## ! Unoptimised rehne de abhi ke liye


def remcount():
    with open(student_record, "rb") as F:
        val = pickle.load(F)
        count = 0
        weak_students = []

        for student in val:
            if student[2] <= 40:
                print(f"{student} eligible for remedial")
                weak_students.append(student)
                count += 1
        print(f"the total number of weak students are {count}")
        print(f"The weak students are {weak_students}")

        # ! highest marks is the key here first marks


def firstmark():
    with open(student_record, "rb") as F:
        val = pickle.load(F)
        count = 0
        main = [i[2] for i in val]

        top = max(main)
        print(top, "is the first mark")

        for i in val:
            if top == i[2]:
                print(f"{i}\ncongrats")
                count += 1
        print("The total number of students who secured top marks are", count)


remcount()
firstmark()
