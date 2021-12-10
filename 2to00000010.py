# this uses GPL V3 LICENSE
# code by @JymPatel

import sys

binary = '$' # just starting var
n = 15 # can get 2**16 numbers


input = int(input("What is your Decimal Num
# main algorithm
while n >= 0:
    if input < 2**n:
        binary = binary + '0'
    else:
        binary = binary + '1'
        input = input - 2**n
    n = n - 1

# get it at https://github.com/JymPatel/Python3-FirstEdition
print("get it at https://github.com/JymPatel/Python3-FirstEdition")
