'''Write a function in python to count the number of lowercase
alphabets present in a text file “happy.txt'''


def lowercase():
    F=open("happy.txt","r")
    count=0
    count_=0
    value=F.read()
    for i in value:
        if i.islower():
            count+=1
        elif i.isupper():
            count_+=1
    print("The total number of lower case letters are",count)
    print("The total number of upper case letters are",count_)
    print("The total number of letters are",count+count_)

lowercase()    
