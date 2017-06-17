#Made on May 27th, 2017
#Made by SlimxShadyx
#Editted by CaptMcTavish, June 17th, 2017

#Dice Rolling Simulator

import random

global user_exit_checker
user_exit_checker="exit"

def start():
    print "Welcome to dice rolling simulator: \nPress Enter to proceed"
    raw_input(">")

    result()

def bye():
    print "Thanks for using the Dice Rolling Simulator! Have a great day! =)"

def result():

    user_dice_chooser


    print "\r\nGreat! Begin by choosing a die! [6] [8] [12]?\r\n" 
    user_dice_chooser = raw_input(">")

    user_dice_chooser = int(user_dice_chooser)

    if user_dice_chooser == 6:
        dice6()

    elif user_dice_chooser == 8:
        dice8()

    elif user_dice_chooser == 12:
        dice12()

    else:
        print "\r\nPlease choose one of the applicable options!\r\n"
        result()


def dice6():
    dice_6 = random.randint(1,6)
    print "\r\nYou rolled a " + str(dice_6) + "!\r\n"

    user_exit_checker_raw = raw_input("\r\nIf you want to roll another die, type [roll]. To exit, type [exit].\r\n?>")
    user_exit_checker = (user_exit_checker_raw.lower())
    if user_exit_checker=="roll":
        start()
    else:
        bye()

def dice8():
    dice_8 = random.randint(1,8)
    print "\r\nYou rolled a " + str(dice_8) + "!"

    user_exit_checker_raw = raw_input("\r\nIf you want to roll another die, type [roll]. To exit, type [exit].\r\n?>")
    user_exit_checker = (user_exit_checker_raw.lower())
    if user_exit_checker=="roll":
        start()
    else:
        bye()

def dice12():
    dice_12 = random.randint(1,12)
    print "\r\nYou rolled a " + str(dice_12) + "!"

    user_exit_checker_raw = raw_input("\r\nIf you want to roll another die, type [roll]. To exit, type [exit].\r\n?>")
    user_exit_checker = (user_exit_checker_raw.lower())
    if user_exit_checker=="roll":
        start()
    else:
        bye()
start()
