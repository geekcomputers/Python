# Python Program to Print Even Numbers from 1 to N

maximum = int(input(" Please Enter the Maximum Value : "))

number = 1

while number <= maximum:
    if number % 2 == 0:
        print(f"{number}")
    number = number + 1
