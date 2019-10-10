# INPUT NUMBER OF EVEN NUMBERS
def print_error_messages():  # function to print error message if user enters a negative number
    print("Invalid number, please enter a Non-negative number!")
    exit()


try:
    n = int(input('Amount: '))  # user input
except ValueError:
    print_error_messages()

start = 0

if n < 0:
    print_error_messages()

for i in range(n):  # loop till the number n entered by the user
    print(start)  # prints 0 first as it is an even number
    start += 2  # increases the value of start by 2 to get to the next even number
