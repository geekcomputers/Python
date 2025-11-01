import logging
import os
import pickle

from dotenv import load_dotenv

base = os.path.dirname(__file__)
load_dotenv(os.path.join(base, ".env"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
student_record = os.getenv("STUDENTS_RECORD_FILE")


def b_read():
    # Opening a file & loading it
    if not os.path.exists(student_record):
        logging.warning("File not found")
        return

    with open(student_record, "rb") as F:
        student = pickle.load(F)
        logging.info("File opened successfully")
        logging.info("Records in the file are:")
        for i in student:
            logging.info(i)


def b_modify():
    # Deleting the Roll no. entered by user
    if not os.path.exists(student_record):
        logging.warning("File not found")
        return
    roll_no = int(input("Enter the Roll No. to be deleted: "))
    student = 0
    with open(student_record, "rb") as F:
        student = pickle.load(F)

    with open(student_record, "wb") as F:
        rec = [i for i in student if i[0] != roll_no]
        pickle.dump(rec, F)


b_read()
b_modify()
