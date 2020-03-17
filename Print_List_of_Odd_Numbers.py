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

result = "0\n"  # Number 0 is added to string
if n == 0:
    print(result)  # If number is equal to zero, output will be just "0"
elif n > 0:
    result += ''.join(str(i)+"\n" for i in range(1, n+1, 2))  # creating string with number "i"
    print(result)                                             # in range from 1 till "n+1" with step 2;
