# Updating records in a binary file

import pickle


def update():
    F = open("class.dat", "rb+")
    S = pickle.load(F)
    found = 0
    rno = int(input("enter the roll number you want to update"))
    for i in S:
        if rno == i[0]:
            print("the currrent name is", i[1])
            i[1] = input("enter the new name")
            found = 1
            break

    if found == 0:
        print("Record not found")

    else:
        F.seek(0)
        pickle.dump(S, F)

    F.close()


update()

F = open("class.dat", "rb")
val = pickle.load(F)
print(val)
F.close()
