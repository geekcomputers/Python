import pickle
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def b_read():
    # Opening a file & loading it
    if not os.path.exists("student_records.pkl"):
        logging.warning("File not found")
        return

    with open("student_records.pkl", "rb") as F:
        student = pickle.load(F)
        logging.info("File opened successfully")
        logging.info("Records in the file are:")
        for i in student:
            logging.info(i)

def b_modify():
    # Deleting the Roll no. entered by user
    if not os.path.exists("student_records.pkl"):
        logging.warning("File not found")
        return
    roll_no = int(input("Enter the Roll No. to be deleted: "))
    student = 0
    with open("student_records.pkl", "rb") as F:
        student = pickle.load(F)

    with open("student_records.pkl", "wb") as F:
        rec = [i for i in student if i[0] != roll_no]
        pickle.dump(rec, F)



b_read()
b_modify()