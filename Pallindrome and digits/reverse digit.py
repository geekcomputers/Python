'''
Created by George Rahul 22/10/2020
123 is 1*100+2*10+3*1.
so, reversing it means 321 which is 3*100+2*10+1*1
'''
import time
x = input("Type the number to reversed:")

n = 1
rev = 0

for i in x:
    i = int(i)
    z = i * n
    n = n * 10
    rev = rev + z

print(rev)

time.sleep(5)
