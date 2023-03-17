""" author: Ataba29
    the code is just a russian roulette game against
    the computer
"""
from random import randrange
import time


def main():

    # create the gun and set the bullet
    numOfRounds = 6
    gun = [0, 0, 0, 0, 0, 0]
    bullet = randrange(0, 6)
    gun[bullet] = 1
    player = False  # is player dead
    pc = False  # is pc dead

    # menu
    print("/********************************/")
    print("    Welcome to russian roulette")
    print("/********************************/")
    time.sleep(2)
    print("you are going to play against the pc")
    time.sleep(2)
    print("there is one gun and one bullet")
    time.sleep(2)
    print("all you have to do is pick who starts first")
    time.sleep(2)

    # take input from the user
    answer = input(
        "please press 'm' if you want to start first or 'p' if you want the pc to start first: "
    )

    # check input
    while answer != "m" and answer != "p":
        answer = input("please enter again ('m' or 'p'): ")

    # set turn
    if answer == 'm':
        turn = "player"
    else:
        turn = "pc"

    # game starts
    while numOfRounds != 0 and (pc == False and player == False):
        print(f"\nRound number {numOfRounds}/6")
        time.sleep(1)
        print("the gun is being loaded")
        time.sleep(3)
        print("the gun is placed on " + ("your head" if turn ==
              "player" else "the cpu of the pc"))
        time.sleep(3)
        print("and...")
        time.sleep(1)
        print("...")
        time.sleep(2)
        print("...")
        time.sleep(2)
        print("...")
        time.sleep(2)

        # get the bullet in the chamber
        shot = gun.pop(numOfRounds - 1)

        if shot:
            print("THE GUN WENT OFF!!!")
            print("YOU DIED" if turn == "player" else "THE PC DIED")
            if turn == "player":  # set up who died
                player = True
            else:
                pc = True
        else:
            print("nothing happened phew!")
            if turn == "player":  # flip the turn
                turn = "pc"
            else:
                turn = "player"

        time.sleep(2)
        numOfRounds -= 1

    time.sleep(1)
    print("")
    if player:
        print("sorry man you died better luck next time")
        print("don't forget to send a pic from heaven :)")
    else:
        print("good job man you survived")
        print("you just got really lucky")
    print("anyways hope you had fun because i sure did")


main()
