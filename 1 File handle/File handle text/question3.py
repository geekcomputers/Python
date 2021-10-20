"""Write a user-defined function named count() that will read
the contents of text file named “happy.txt” and count
the number of lines which starts with either “I‟ or “M‟."""

#step1:
def write_to_file():
    F=open("happy.txt","a")
    while True:
        text=input("enter any text")
       
        F.write(text+"\n") #write function takes exactly 1 arguement so concatenation
        choice= input("do you want to enter more, y/n")
        if choice=="n":
            break
    F.close()    
#write_to_file()

#step2:
def check_first_letter():
    F=open("happy.txt","r")
    value=F.read()
    count=0
    line=value.split()
    for i in line:
        if i[0] in ["m","M","i","I"]:
            count+=1
            print(i)
            
    print("The total number of sentences starting with I or M are",count)        
    
check_first_letter()    
        
        
        
