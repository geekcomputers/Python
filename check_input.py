def get_user_input(start,end):

    """
        input: two integer values
               lower limit 'start' and maximum 'end'
               the arguments aren't inclusive.

        output: if reading successful then returns the read integer. 

        purpose: reads from command-line a integer in the given bounds. 
                 while input invalid asks user again
    """

    loop = True # controls while-loop

    while (loop):

        try:

            # reads and converts the input from the console. 
            userInput = int(input("Enter Your choice: "))

            # checks whether input is in the given bounds.
            if userInput > end  or userInput < start:

                # error case
                print("Please try again. Not in valid bounds.")

            else:

                # valid case
                loop = False # aborts while-loop

        except ValueError:

            # error case
            print("Please try again. Only numbers") 

    return userInput
        

x = get_user_input(1,6)
print(x)
###Asks user to enter something, ie. a number option from a menu.
###While type != interger, and not in the given range,
###Program gives error message and asks for new input.
