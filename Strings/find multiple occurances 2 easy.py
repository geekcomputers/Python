str = input("Type the string")
ch=input("Type the occurance needed to be found:")
i=0


while i <len(str):
    start=i
    ins=str.find(ch,i,len(str))
    if ins==-1:
        break
    else:
        print(ins)
        i=ins+1