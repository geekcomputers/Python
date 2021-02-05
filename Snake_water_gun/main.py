import random
import time

class bcolors:
    HEADERS = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[93m'
    WARNING = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

run = True
li = ["s", "w", "g"]

print(bcolors.OKBLUE + bcolors.BOLD + "Welcome to the game 'Snake-Water-Gun'.\nWanna play? Type Y or N" + bcolors.ENDC)
b = input()
if b == "N" or b=="n":
    run = False
    print("Ok bubyeee! See you later")
elif b=="Y" or b=="y":
    print("There will be 10 matches and the one who won max matches will win Let's start")

i=0
w=0
l=0
while run and i<10:
    c = random.choice(li)
    a = input("Type s for snake,w for water or g for gun: ")

    if a==c:
        print(bcolors.HEADERS + "Game draws.Play again" + bcolors.ENDC)
        i+=1
        print(f"{10-i} matches left")

    elif a=="s" and c=="g":
        print(bcolors.FAIL +"It's Snake v/s Gun You lose!" + bcolors.ENDC)
        i+=1
        l+=1
        print(f"{10-i} matches left")
    elif a=="s" and c=="w":
        print(bcolors.OKGREEN + "It's Snake v/s Water. You won" + bcolors.ENDC)
        i += 1
        w += 1 
        print(f"{10-i} matches left")

    elif a=="w" and c=="s":
        print(bcolors.FAIL +"It's Snake v/s Gun You lose!" + bcolors.ENDC)
        i+=1
        l+=1
        print(f"{10-i} matches left")
    elif a=="w" and c=="g":
        print(bcolors.OKGREEN + "It's Snake v/s Water. You won" + bcolors.ENDC)
        i += 1
        w += 1
        print(f"{10-i} matches left")

    elif a=="g" and c=="w":
        print(bcolors.FAIL +"It's Snake v/s Gun You lose!" + bcolors.ENDC)
        i+=1
        l+=1
        print(f"{10-i} matches left")
    elif a=="g" and c=="s":
        print(bcolors.OKGREEN + "It's Snake v/s Water. You won" + bcolors.ENDC)
        i += 1
        w += 1
        print(f"{10-i} matches left")

    else:
        print("Wrong input")

if run == True:
    print(f"You won {w} times and computer won {l} times.Final result is...")
    time.sleep(3)
    if w >= l:
        print(bcolors.OKGREEN + bcolors.BOLD + "Woooh!!!!!!! Congratulations you won" + bcolors.ENDC)
    elif l==w:
        print("Game draws!!!!!!!")
    elif w <= l:
        print(bcolors.FAIL + bcolors.BOLD + "You lose!!!.Better luck next time" + bcolors.ENDC)



