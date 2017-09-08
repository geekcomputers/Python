def get_user_input(start,end):

    while (1):
        try:
            userInput = int(input("Enter Your choice: "))
            if userInput > end  or userInput < start:
                print("Please try again.")
            else:
                return userInput

        except ValueError:
            print("Please try again.") 
        

x = get_user_input(1,6)
print(x)
###Asks user to enter something, ie. a number option from a menu.
###While type != interger, and not in the given range,
###Program gives error message and asks for new input.
