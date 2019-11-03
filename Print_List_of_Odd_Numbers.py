# INPUT NUMBER OF ODD NUMBERS
def print_error_messages():
    print("Invalid number, please enter a Non-negative number!")
    exit()


try:
    n = int(input("Amount: "))
except ValueError:
    print_error_messages()

start = 1

if n < 0:
    print_error_messages()

for i in range(n):
    if n == 0:
        start += 2  # increases the value of start by 2 to get to the next even number
        print(start)  # prints 0 first as it is an even number
        start += 2  # increases the value of start by 2 to get to the next even number
    elif n % 2 == 0:
        print(start)
