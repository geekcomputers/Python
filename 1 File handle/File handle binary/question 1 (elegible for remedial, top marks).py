"""Amit is a monitor of class XII-A and he stored the record of all
the students of his class in a file named “class.dat”.
Structure of record is [roll number, name, percentage]. His computer
teacher has assigned the following duty to Amit

Write a function remcount( ) to count the number of students who need
 remedial class (student who scored less than 40 percent)

 
 """
# also find no. of children who got top marks

import pickle

list = [
    [1, "Ramya", 30],
    [2, "vaishnavi", 60],
    [3, "anuya", 40],
    [4, "kamala", 30],
    [5, "anuraag", 10],
    [6, "Reshi", 77],
    [7, "Biancaa.R", 100],
    [8, "sandhya", 65],
]

with open("class.dat", "ab") as F:
    pickle.dump(list, F)
    F.close()


def remcount():
    with open("class.dat", "rb") as F:
        val = pickle.load(F)
        count = 0

        for i in val:
            if i[2] <= 40:
                print(f"{i} eligible for remedial")
                count += 1
        print(f"the total number of students are {count}")


remcount()


def firstmark():
    with open("class.dat", "rb") as F:
        val = pickle.load(F)
        count = 0
        main = [i[2] for i in val]

        top = max(main)
        print(top, "is the first mark")

        F.seek(0)
        for i in val:
            if top == i[2]:
                print(f"{i}\ncongrats")
                count += 1

        print("the total number of students who secured top marks are", count)


firstmark()

with open("class.dat", "rb") as F:
    val = pickle.load(F)
    print(val)
