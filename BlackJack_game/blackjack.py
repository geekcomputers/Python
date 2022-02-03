# master
# master
# BLACK JACK - CASINO A GAME OF FORTUNE!!!
from time import *

# BLACK JACK - CASINO
# PYTHON CODE BASE


# master
import random

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4

random.shuffle(deck)

print(f'{"*"*58} \n Welcome to the game Casino - BLACK JACK ! \n{"*"*58}')
sleep(2)
print("So Finally You Are Here To Accept Your Fate")
sleep(2)
print("I Mean Your Fortune")
sleep(2)
print("Lets Check How Lucky You Are  Wish You All The Best")
sleep(2)
print("Loading---")
sleep(2)

print("Still Loading---")
sleep(2)
print(
    "So You Are Still Here Not Gone I Gave You Chance But No Problem May Be You Trust Your Fortune A Lot \n Lets Begin Then"
)
sleep(2)
d_cards = []  # Initialising dealer's cards
p_cards = []  # Initialising player's cards
sleep(2)
while len(d_cards) != 2:
    random.shuffle(deck)
    d_cards.append(deck.pop())
    if len(d_cards) == 2:
        print("The cards dealer has are X ", d_cards[1])

# Displaying the Player's cards
while len(p_cards) != 2:
    random.shuffle(deck)
    p_cards.append(deck.pop())
    if len(p_cards) == 2:
        print("The total of player is ", sum(p_cards))
        print("The cards Player has are  ", p_cards)

if sum(p_cards) > 21:
    print(f"You are BUSTED !\n  {'*'*14}Dealer Wins !!{'*'*14}\n")
    exit()

if sum(d_cards) > 21:
    print(f"Dealer is BUSTED !\n   {'*'*14} You are the Winner !!{'*'*18}\n")
    exit()

if sum(d_cards) == 21:
    print(f"{'*'*24}Dealer is the Winner !!{'*'*14}")
    exit()

if sum(d_cards) == 21 and sum(p_cards) == 21:
    print(f"{'*'*17}The match is tie !!{'*'*25}")
    exit()


# function to show the dealer's choice
def dealer_choice():
    if sum(d_cards) < 17:
        while sum(d_cards) < 17:
            random.shuffle(deck)
            d_cards.append(deck.pop())

    print("Dealer has total " + str(sum(d_cards)) + "with the cards ", d_cards)

    if sum(p_cards) == sum(d_cards):
        print(f"{'*'*15}The match is tie !!{'*'*15}")
        exit()

    if sum(d_cards) == 21:
        if sum(p_cards) < 21:
            print(f"{'*'*23}Dealer is the Winner !!{'*'*18}")
        elif sum(p_cards) == 21:
            print(f"{'*'*20}There is tie !!{'*'*26}")
        else:
            print(f"{'*'*23}Dealer is the Winner !!{'*'*18}")

    elif sum(d_cards) < 21:
        if sum(p_cards) < 21 and sum(p_cards) < sum(d_cards):
            print(f"{'*'*23}Dealer is the Winner !!{'*'*18}")
        if sum(p_cards) == 21:
            print(f"{'*'*22}Player is winner !!{'*'*22}")
        if 21 > sum(p_cards) > sum(d_cards):
            print(f"{'*'*22}Player is winner !!{'*'*22}")

    else:
        if sum(p_cards) < 21:
            print(f"{'*'*22}Player is winner !!{'*'*22}")
        elif sum(p_cards) == 21:
            print(f"{'*'*22}Player is winner !!{'*'*22}")
        else:
            print(f"{'*'*23}Dealer is the Winner !!{'*'*18}")


while sum(p_cards) < 21:

    # to continue the game again and again !!
    k = input("Want to hit or stay?\n Press 1 for hit and 0 for stay ")
    if k == 1:
        random.shuffle(deck)
        p_cards.append(deck.pop())
        print("You have a total of " + str(sum(p_cards)) + " with the cards ", p_cards)
        if sum(p_cards) > 21:
            print(f'{"*"*13}You are BUSTED !{"*"*13}\n Dealer Wins !!')
        if sum(p_cards) == 21:
            print(f'{"*"*19}You are the Winner !!{"*"*29}')

    else:
        dealer_choice()
        break
