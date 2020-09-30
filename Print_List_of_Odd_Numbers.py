# CALCULATE NUMBER OF ODD NUMBERS

n = int(input("Enter the limit : "))  # user input

if n <= 0:
    print("Invalid number, please enter a number greater than zero!")
else:    
    odd_list = [i for i in range(1,n+1,2)]      # creating string with number "i"
    print(odd_list)                             # in range from 1 till "n".
