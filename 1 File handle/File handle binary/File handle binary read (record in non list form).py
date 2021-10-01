import pickle

def binaryread():
    B=open("studrec.dat","rb")
    stud=pickle.load(B)
    print(stud)

    # prints the whole record in nested list format
    print("contents of binary file")

    for ch in stud:
        
        print(ch) #prints one of the chosen rec in list
        
        Rno=ch[0]
        Rname=ch[1]  #due to unpacking the val not printed in list format
        Rmark=ch[2]

        print(Rno, Rname,Rmark, end="\t")
    
        B.close

binaryread()
