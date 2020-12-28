str = input("Type the string:")
ch = input("Type the character/word you want to find the occurance:")
start=0
n=len(str)
count=set()
for i in range(0,n+1):
    x=str.find(ch,i)

    if x !=-1:
        count.add(x)


if len(count)==0:
    print("Sorry no occurances found")
else:

  print("The index of the occurance is/are",count,"repeating",len(count),"times")
