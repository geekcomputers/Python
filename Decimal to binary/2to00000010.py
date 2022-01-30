# this uses GPL V3 LICENSE
# code by @JymPatel

import sys

binary = "$"  # just starting var
n = 15  # can get 2**16 numbers

# get integer as output which is less than limit 2**16
try:
    input = int(input("What is your Decimal Number?"))
    limit = 2 ** (n + 1)
    input <= limit
except ValueError:
    print("Please put integer in input! & less than", limit)
    sys.exit()

# main algorithm
while n >= 0:
    if input < 2 ** n:
        binary = binary + "0"
    else:
        binary = binary + "1"
        input = input - 2 ** n
    n = n - 1

print(binary)

# get it at https://github.com/JymPatel/Python3-FirstEdition
print("get it at https://github.com/JymPatel/Python3-FirstEdition")
