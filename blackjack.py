# BLACK JACK - CASINO

import random

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4


def welcome():
    print(
        "                       **********************************************************                                    "
    )
    print(
        "                                   Welcome to the game Casino - BLACK JACK !                                         "
    )
    print(
        "                       **********************************************************                                    "
    )


def start_game():
    random.shuffle(deck)

    d_cards = []
    p_cards = []

    # Dealer initial cards
    while len(d_cards) != 2:
        random.shuffle(deck)
        d_cards.append(deck.pop())
        if len(d_cards) == 2:
            print("The cards dealer has are X ", d_cards[1])

    # Player initial cards
    while len(p_cards) != 2:
        random.shuffle(deck)
        p_cards.append(deck.pop())
        if len(p_cards) == 2:
            print("The total of player is ", sum(p_cards))
            print("The cards Player has are  ", p_cards)

    if sum(p_cards) > 21:
        print("You are BUSTED !\n  **************Dealer Wins !!******************\n")
        return

    if sum(d_cards) > 21:
        print(
            "Dealer is BUSTED !\n   ************** You are the Winner !!******************\n"
        )
        return

    if sum(d_cards) == 21 and sum(p_cards) == 21:
        print("*****************The match is tie !!*************************")
        return

    if sum(d_cards) == 21:
        print("***********************Dealer is the Winner !!******************")
        return

    def dealer_choice():
        if sum(d_cards) < 17:
            while sum(d_cards) < 17:
                random.shuffle(deck)
                d_cards.append(deck.pop())

        print("Dealer has total " + str(sum(d_cards)) + " with the cards ", d_cards)

        if sum(p_cards) == sum(d_cards):
            print("***************The match is tie !!****************")
            return

        if sum(d_cards) > 21:
            print("**********************Player is winner !!**********************")
            return

        if sum(d_cards) > sum(p_cards):
            print("***********************Dealer is the Winner !!******************")
        else:
            print("**********************Player is winner !!**********************")

    # Player turn
    while sum(p_cards) < 21:
        k = input("Want to hit or stay?\n Press 1 for hit and 0 for stay ")

        if k == "1":
            random.shuffle(deck)
            p_cards.append(deck.pop())
            print("You have a total of " + str(sum(p_cards)) + " with the cards ", p_cards)

            if sum(p_cards) > 21:
                print("*************You are BUSTED !*************\n Dealer Wins !!")
                return

            if sum(p_cards) == 21:
                print(
                    "*******************You are the Winner !!*****************************"
                )
                return
        else:
            dealer_choice()
            break


# Run Game
welcome()
start_game()
