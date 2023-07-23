# This is an edited version
# Made the code much more easier to read
# Used better naming for variables
# There were few inconsistencies in the outputs of the first if/else/if ladder \
# inside the while loop. That is solved.
import random
import time
from os import system


class bcolors:
    HEADERS = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[93m"
    WARNING = "\033[92m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


run = True
li = ["s", "w", "g"]

system("clear")
b = input(
    bcolors.OKBLUE
    + bcolors.BOLD
    + "Welcome to the game 'Snake-Water-Gun'.\nWanna play? Type Y or N: "
    + bcolors.ENDC
).capitalize()

if b == "N":
    run = False
    print("Ok bubyeee! See you later")
elif b == "Y" or b == "y":
    print(
        "There will be 10 matches, and the one who wins more matches will win. Let's start."
    )

i = 0
score = 0

while run and i < 10:

    comp_choice = random.choice(li)
    user_choice = input("Type s for snake, w for water or g for gun: ").lower()

    if user_choice == comp_choice:
        print(bcolors.HEADERS + "Game draws. Play again" + bcolors.ENDC)

    elif user_choice == "s" and comp_choice == "g":
        print(bcolors.FAIL + "It's Snake v/s Gun You lose!" + bcolors.ENDC)

    elif user_choice == "s" and comp_choice == "w":
        print(bcolors.OKGREEN + "It's Snake v/s Water. You won" + bcolors.ENDC)
        score += 1

    elif user_choice == "w" and comp_choice == "s":
        print(bcolors.FAIL + "It's Water v/s Snake You lose!" + bcolors.ENDC)

    elif user_choice == "w" and comp_choice == "g":
        print(bcolors.OKGREEN + "It's Water v/s Gun. You won" + bcolors.ENDC)
        score += 1

    elif user_choice == "g" and comp_choice == "w":
        print(bcolors.FAIL + "It's Gun v/s Water You lose!" + bcolors.ENDC)

    elif user_choice == "g" and comp_choice == "s":
        print(bcolors.OKGREEN + "It's Gun v/s Snake. You won" + bcolors.ENDC)
        score += 1

    else:
        print("Wrong input")
        continue

    i += 1
    print(f"{10-i} matches left")

if run == True:
    print(f"Your score is {score} and the final result is...")
    time.sleep(3)
    if score > 5:
        print(
            bcolors.OKGREEN
            + bcolors.BOLD
            + "Woooh!!!!!!! Congratulations you won"
            + bcolors.ENDC
        )
    elif score == 5:
        print("Game draws!!!!!!!")
    elif score < 5:
        print(
            bcolors.FAIL
            + bcolors.BOLD
            + "You lose!!!. Better luck next time"
            + bcolors.ENDC
        )
