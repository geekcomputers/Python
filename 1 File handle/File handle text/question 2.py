""" Write a method/function DISPLAYWORDS() in python to read lines
 from a text file STORY.TXT,
 using read function
and display those words, which are less than 4 characters. """

F=open("story.txt","r")
value=F.read()
lines=value.split()
count=0

for i in lines:
    if len(i)<4:
        print(i)
        count+=1
    else:
        pass

print("The total number of words with length less than 4 are",count)
