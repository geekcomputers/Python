"""
This is a snake water gun game similar to rock paper scissor
In this game :
if computer chooses snake and user chooses water, the snake will drink water and computer wins.
If computer chooses gun and user chooses water, the gun gets drown into water and user wins.
And so on for other cases
"""

# you can use this code also, see this code is very short in compare to your code
# code starts here
"""
# Snake || Water || Gun __ Game
import random
times = 10 # times to play game
comp_choice = ["s","w","g"] # output choice for computer
user_point = 0 # user point is initially marked 0
comp_point = 0 # computer point is initially marked 0
while times >= 1:
    comp_rand = random.choice(comp_choice) # output computer will give
    #
    # print(comp_rand) # checking if the code is working or not
    print(f"ROUND LEFT = {times}")
# checking if the input is entered correct or not
    try:
        user_choice = input("Enter the input in lowercase ex. \n (snake- s) (water- w) (gun- w)\n:- ") # user choice, the user will input
    except Exception as e:
        print(e)
# if input doen't match this will run
    if user_choice != 's' and user_choice != 'w' and user_choice != 'g':
            print("Invalid input, try again\n")
            continue
# checking the input and calculating score
    if comp_rand == 's':
        if user_choice == 'w':
            comp_point += 1
        elif user_choice == 'g':
            user_point += 1

    elif comp_rand == 'w':
        if user_choice == 'g':
            comp_point += 1
        elif user_choice == 's':
            user_point += 1

    elif comp_rand == 'g':
        if user_choice == 's':
            comp_point += 1
        elif user_choice == 'w':
            user_point += 1

    times -=1 # reducing the number of rounds after each match
if user_point>comp_point: # if user wins
    print(f"WOOUUH! You have win \nYour_point = {user_point}\nComputer_point = {comp_point}")
elif comp_point>user_point: # if computer wins
    print(f"WE RESPECT YOUR HARD WORK, BUT YOU LOSE AND YOU ARE A LOSER NOW! \nYour_point = {user_point}\nComputer_point = {comp_point}")
elif comp_point==user_point: # if match draw
    print(f"MATCH DRAW\nYour_point = {user_point}\nComputer_point = {comp_point}")
else: # just checked
    print("can't calculate score")
exit = input("PRESS ENTER TO EXIT")
"""  # code ends here
import random

# import time

choices = {"S": "Snake", "W": "Water", "G": "Gun"}

x = 0
comp_point = 0
user_point = 0
match_draw = 0

print("Welcome to the Snake-Water-Gun Game\n")
print("I am Mr. Computer, We will play this game 10 times")
print("Whoever wins more matches will be the winner\n")

while x < 10:
    print(f"Game No. {x+1}")
    for key, value in choices.items():
        print(f"Choose {key} for {value}")

    comp_rand = random.choice(list(choices.keys())).lower()
    user_choice = input("\n----->").lower()
    print("Mr. Computer's choice is : " + comp_rand)

    # you can use this code to minimize your writing time for the code
    """
    if comp_rand == 's':
        if user_choice == 'w':
            print("\n-------Mr. Computer won this round--------")
            comp_point += 1
        elif user_choice == 'g':
            print("\n-------You won this round-------")
            user_point += 1
        else:
            match_draw +=1

    elif comp_rand == 'w':
        if user_choice == 'g':
            print("\n-------Mr. Computer won this round--------")
            comp_point += 1
        elif user_choice == 's':
            print("\n-------You won this round-------")
            user_point += 1
        else:
            match_draw +=1

    elif comp_rand == 'g':
        if user_choice == 's':
            print("\n-------Mr. Computer won this round--------")
            comp_point += 1
        elif user_choice == 'w':
            print("\n-------You won this round-------")
            user_point += 1
        else:
            match_draw +=1

    """

    if comp_rand == "s":
        if user_choice == "w":
            print("\n-------Mr. Computer won this round--------")
            comp_point += 1
            x += 1
        elif user_choice == "g":
            print("\n-------You won this round-------")
            user_point += 1
            x += 1
        else:
            print("\n-------Match draw-------")
            match_draw += 1
            x += 1

    elif comp_rand == "w":
        if user_choice == "g":
            print("\n-------Mr. Computer won this round--------")
            comp_point += 1
            x += 1
        elif user_choice == "s":
            print("\n-------You won this round-------")
            user_point += 1
            x += 1
        else:
            print("\n-------Match draw-------")
            match_draw += 1
            x += 1

    elif comp_rand == "g":
        if user_choice == "s":
            print("\n-------Mr. Computer won this round--------")
            comp_point += 1
            x += 1
        elif user_choice == "w":
            print("\n-------You won this round-------")
            user_point += 1
            x += 1
        else:
            print("\n-------Match draw-------")
            match_draw += 1
            x += 1

print("Here are final stats of the 10 matches : ")
print(f"Mr. Computer won : {comp_point} matches")
print(f"You won : {user_point} matches")
print(f"Matches Drawn : {match_draw}")

if comp_point > user_point:
    print("\n-------Mr. Computer won-------")

elif comp_point < user_point:
    print("\n-----------You won-----------")

else:
    print("\n----------Match Draw----------")
