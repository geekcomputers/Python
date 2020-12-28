import time
x=int(input("Type the number to check:"))
y=x
rev = 0
while x!=0:
    d=x%10 #takes the reminder
    rev=rev*10+d
    x=x//10 #takes the quotient



if y==rev:
    print(f"This number is a pallindrome")
else:
    print("This number is not a pallindrome")

time.sleep(5)