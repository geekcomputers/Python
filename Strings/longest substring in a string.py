import time

x = input("Enter the string:   ")

y = x.split()

z = 0
ab = ''

for i in y:
    l = len(i)

    if l > z:
        z = l
        ab = i

print("The longest substring is", ab, "with a length of ", z)
time.sleep(3)
