import pickle

def Bdelete():
    # Opening a file & loading it
    F= open("studrec.dat","rb")
    stud = pickle.load(F)
    F.close()
    
    print(stud)
    
    # Deleting the Roll no. entered by user
    rno= int(input("Enter the Roll no. to be deleted: "))
    F= open("studrec.dat","wb")
    rec= []
    for i in stud:
        if i[0] == rno:
            continue
        rec.append(i)
    pickle.dump(rec,F)
    F.close()
    
Bdelete()
