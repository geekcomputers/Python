str = input("Type the string:")
ch = input("Type the character you want to find:")
count = 0

for i in str:
    if i == ch:
        count +=1

print("The number of time",ch,"repeated is",count)