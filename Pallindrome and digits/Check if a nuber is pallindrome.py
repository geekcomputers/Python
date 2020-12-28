import time
x = input("Type the number to check if it is a pallindrome:")

n = 1
rev = 0

for i in x:
    i = int(i)
    z = i * n
    n = n * 10
    rev = rev + z

rev=f"{rev}"
if rev==x:
    print(f"{x} is a pallindrome")
else:
    print(f"{x} is not a pallindrome")