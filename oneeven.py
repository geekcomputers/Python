# Python Program to Print Even Numbers from 1 to N

maximum = int(input(" Please Enter the Maximum Value : "))

for number in range(1, maximum + 1):
    if number % 2 == 0:
        print("{0}".format(number))
