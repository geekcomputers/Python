import sys

def get_integer_input(prompt, attempts):
    for i in range(attempts, 0, -1):
        try:
            n = int(input(prompt))
            return n
        except ValueError:
            print("Enter an integer only")
            print(f"{i-1} {'chance' if i-1 == 1 else 'chances'} left")
    return None

def sum_of_digits(n):
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total

chances = 3
number = get_integer_input("Enter a number: ", chances)

if number is None:
    print("You've used all your chances.")
    sys.exit()

result = sum_of_digits(number)
print(f"The sum of the digits of {number} is: {result}")
