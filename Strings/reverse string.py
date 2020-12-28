import time

x=input("Enter the string to be reversed:")

z=len(x)
rev=list(x)

for i in range(z-1,-1,-1):
    print(rev[i],end='')

time.sleep(5)