#user can give input now
#python3

import sys

if sys.version_info[0]==2:
    version=2
else:
    version=3 

if version==2:
    n=input('Enter number of even numbers to print: ')
    printed=0
    numbers=0
    while printed!=n:
        if numbers%2==0:
            print numbers,
            printed+=1
        numbers+=1

if version==3:
    print ([x for x in range(int(input()),int(input())) if not x%2])



