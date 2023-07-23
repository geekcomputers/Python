# using codeSkulpter

import random

import simplegui


def new_game():
    global num
    print("new game starts")


def range_of_100():
    global num
    num = random.randrange(0, 100)
    print("your range is 0-100")


def range_of_1000():
    global num
    num = random.randrange(0, 1000)
    print("Your range is 0-1000")


def input_guess(guess):
    global num
    print("Your Guess is ", guess)
    num1 = int(guess)
    if num1 == num:
        print("Correct")
    elif num1 >= num:
        print("Greater")
    elif num1 <= num:
        print("Lower")


frame = simplegui.create_frame("Guess The Number", 200, 200)
frame.add_button("range[0-1000)", range_of_1000)
frame.add_button("range[0-100)", range_of_100)
frame.add_input("enter your guess", input_guess, 200)
frame.start()
new_game()
