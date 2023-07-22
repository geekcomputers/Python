import pickle


def binary_read():
    with open("studrec.dat") as b:
        stud = pickle.load(b)
        print(stud)

        # prints the whole record in nested list format
        print("contents of binary file")

        for ch in stud:

            print(ch)  # prints one of the chosen rec in list

            rno = ch[0]
            rname = ch[1]  # due to unpacking the val not printed in list format
            rmark = ch[2]

            print(rno, rname, rmark, end="\t")


binary_read()
