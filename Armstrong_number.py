def is_armstrong_number(number):
    total = 0

    # find the sum of the cube of each digit
    temp = number
    while temp > 0:
        digit = temp % 10
        total += digit ** 3
        temp //= 10
    
    # return the result
    if number == total:
        return True
    else:
        return False

number = int(input("Enter the number: "))
if is_armstrong_number(number):
    print(number,"is an Armstrong number")
else:
    print(number,"is not an Armstrong number")
