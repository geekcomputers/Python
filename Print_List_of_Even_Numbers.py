# INPUT NUMBER OF EVEN NUMBERS

n = int(input("Enter the limit : "))  # user input

if n < 0:
    print("Invalid number, please enter a Non-negative number!")
else:    
    even_list = [i for i in range(0,n+1,2)]         # creating string with number "i"
    print(even_list)                                # in range from 0 till "n" with step 2
