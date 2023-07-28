# Updating records in a binary file

import pickle


def update():
    with open("class.dat", "rb+") as F:
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

        else:
            F.seek(0)
            pickle.dump(S, F)


update()

with open("class.dat", "rb") as F:
    print(pickle.load(F))
