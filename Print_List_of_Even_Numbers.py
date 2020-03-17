# INPUT NUMBER OF EVEN NUMBERS
def print_error_messages():  # function to print error message if user enters a negative number
    print("Invalid number, please enter a Non-negative number!")
    exit()


n = int(input())  # user input


if n < 0:
    print_error_messages()


result = ''.join(str(i)+"\n" for i in range(0,n+1,2))  # creating string with number "i"
print(result)                                          # in range from 0 till "n" with step 2;
