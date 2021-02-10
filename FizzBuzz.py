# FizzBuzz
# A program that prints the numbers from 1 to num (User given number)!
# For multiples of ‘3’ print “Fizz” instead of the number. 
# For the multiples of ‘5’ print “Buzz”.
# If the number is divisible by both 3 and 5 then print "FizzBuzz".
# If none of the given conditions are true then just print the number!


def FizzBuzz():
    num = int(input("Enter the number here: "))
    for i in range(1, num+1):
        if i%3 == 0 and i%5 == 0:
            print("FizzBuzz")
        elif i%3 == 0:
            print("Fizz")
        elif i%5 == 0:
            print("Buzz")
        else:
            print(i)

FizzBuzz()