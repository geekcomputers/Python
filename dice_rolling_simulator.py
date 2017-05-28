#Made on May 27th, 2017
#Made by SlimxShadyx

#Dice Rolling Simulator

import random

#These variables are used for user input and while loop checking.
correct_word = False
dice_checker = False
dicer = False
roller_loop = False

#Checking the user input to start the program.
while correct_word == False:

    user_input_raw = raw_input("\r\nWelcome to the Dice Rolling Simulator! We currently support 6, 8, and 12 sided die! Type [start] to begin!\r\n?>")

    #Converting the user input to lower case.
    user_input = (user_input_raw.lower())

    if user_input == 'start':
        correct_word = True
    
    else:
        print "Please type [start] to begin!\r\n"

#Main program loop. Exiting this, exits the program.
while roller_loop == False:

    #Second While loop to ask the user for the certain die they want.
    while dice_checker == False:
        user_dice_chooser = raw_input("\r\nGreat! Begin by choosing a die! [6] [8] [10]\r\n?>")

        user_dice_chooser = int(user_dice_chooser)

        if user_dice_chooser == 6:
            dice_checker = True

        elif user_dice_chooser == 8:
            dice_checker = True

        elif user_dice_chooser == 12:
            dice_checker = True

        else:
            print "\r\nPlease choose one of the applicable options!\r\n"
    
    #Another inner while loop. This one does the actual rolling, as well as allowing the user to re-roll without restarting the program.
    while dicer == False:

        if user_dice_chooser == 6:
            dice_6 = random.randint(1,6)
            print "\r\nYou rolled a " + str(dice_6) + "!\r\n"
            dicer = True

            user_exit_checker_raw = raw_input("\r\nIf you want to roll another die, type [roll]. To exit, type [exit].\r\n?>")
            user_exit_checker = (user_exit_checker_raw.lower())

            if user_exit_checker == 'roll':
                dicer = False

            elif user_exit_checker == 'exit':
                roller_loop = True


        elif user_dice_chooser == 8:
            dice_8 = random.randint(1,8)
            print "\r\nYou rolled a " + str(dice_8) + "!"
            dicer = True

            user_exit_checker_raw = raw_input("\r\nIf you want to roll another die, type [roll]. To exit, type [exit].\r\n?>")
            user_exit_checker = (user_exit_checker_raw.lower())

            if user_exit_checker == 'roll':
                dicer = False

            elif user_exit_checker == 'exit':
                roller_loop = True

        elif user_dice_chooser == 12:
            dice_12 = random.randint(1,12)
            print "\r\nYou rolled a " + str(dice_12) + "!"
            dicer = True
            
            user_exit_checker_raw = raw_input("\r\nIf you want to roll another die, type [roll]. To exit, type [exit].\r\n?>")
            user_exit_checker = (user_exit_checker_raw.lower())

            if user_exit_checker == 'roll':
                dicer = False

            elif user_exit_checker == 'exit':
                roller_loop = True

print "Thanks for using the Dice Rolling Simulator! Have a great day! =)"
       
