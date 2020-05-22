n = int(input("Enter the required range : "))  # user input
list = []

if (n < 0):
    print("Not a valid number, please enter a positive number!")
else:
    for i in range(0,n+1):
        if(i%2==0):
            list.append(i)          #appending items to the initialised list getting from the 'if' statement

print(list)
