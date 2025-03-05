import pickle


def bdelete():
    # Opening a file & loading it
    with open("studrec.dat","rb") as F:
        stud = pickle.load(F)
        print(stud)

    # Deleting the Roll no. entered by user
    rno = int(input("Enter the Roll no. to be deleted: "))
    with open("studrec.dat","wb") as F:
        rec = [i for i in stud if i[0] != rno]
        pickle.dump(rec, F)


bdelete()
