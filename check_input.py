def get_user_input(start,end):

    testcase = False
    while testcase == False:
        try:
            userInput = int(input("Enter Your choice: "))
            if userInput > 6 or userInput < 1:
                print("Please try again.")
                testcase = False
            else:
                return userInput

        except ValueError:
            print("Please try again.") 
        

x = get_user_input(1,6)
print(x)


###Asks user to enter something, ie. a number option from a menu.
###While type != interger, and not in the given range,
###Program gives error message and asks for new input.
