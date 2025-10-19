#      uno game      #

import random
from typing import List

"""
Generate the UNO deck of 108 cards.

Doctest examples:

>>> deck = buildDeck()
>>> len(deck)
108
>>> sum(1 for c in deck if 'Wild' in c)
8

Return: list of card strings (e.g. 'Red 7', 'Wild Draw Four')
"""


def buildDeck() -> List[str]:
    deck: List[str] = []
    # example card:Red 7,Green 8, Blue skip
    colours = ["Red", "Green", "Yellow", "Blue"]
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Draw Two", "Skip", "Reverse"]
    wilds = ["Wild", "Wild Draw Four"]
    for colour in colours:
        for value in values:
            cardVal = "{} {}".format(colour, value)
            deck.append(cardVal)
            if value != 0:
                deck.append(cardVal)
    for i in range(4):
        deck.append(wilds[0])
        deck.append(wilds[1])
    return deck


"""
Shuffles a list of items passed into it
Parameters: deck=>list
Return values: deck=>list
"""


def shuffleDeck(deck: List[str]) -> List[str]:
    # use Python's built-in shuffle which is efficient and correct
    random.shuffle(deck)
    return deck


"""Draw card function that draws a specified number of cards off the top of the deck
Parameters: numCards -> integer
Return: cardsDrawn -> list
"""


def drawCards(numCards: int) -> List[str]:
    """
    Draw a number of cards from the top of the global `unoDeck`.

    Raises ValueError if the deck runs out of cards.
    """
    cardsDrawn: List[str] = []
    for x in range(numCards):
        try:
            cardsDrawn.append(unoDeck.pop(0))
        except IndexError:
            raise ValueError("The deck is empty; cannot draw more cards")
    return cardsDrawn


"""
Print formatted list of player's hand
Parameter: player->integer , playerHand->list
Return: None
"""


def showHand(player: int, playerHand: List[str]) -> None:
    print("Player {}'s Turn".format(players_name[player]))
    print("Your Hand")
    print("------------------")
    y = 1
    for card in playerHand:
        print("{}) {}".format(y, card))
        y += 1
    print("")


"""
Check whether a player is able to play a card, or not
Parameters: discardCard->string,value->string, playerHand->list
Return: boolean
"""


def canPlay(colour: str, value: str, playerHand: List[str]) -> bool:
    """
    Return True if any card in playerHand is playable on a discard with given colour and value.

    >>> canPlay('Red','5',['Red 3','Green 5'])
    True
    >>> canPlay('Blue','7',['Green 1'])
    False
    """
    for card in playerHand:
        if "Wild" in card:
            return True
        elif colour in card or value in card:
            return True
    return False


# --- Global deck and initial setup ---
unoDeck = buildDeck()
unoDeck = shuffleDeck(unoDeck)
unoDeck = shuffleDeck(unoDeck)
discards: List[str] = []

players_name: List[str] = []
players: List[List[str]] = []
colours = ["Red", "Green", "Yellow", "Blue"]


def main() -> None:
    """Run interactive UNO game (keeps original behavior).

    Note: main() is interactive and not exercised by doctest.
    """
    global players_name, players, discards

    numPlayers = int(input("How many players?"))
    while numPlayers < 2 or numPlayers > 4:
        numPlayers = int(
            input("Invalid. Please enter a number between 2-4.\nHow many players?")
        )
    for player in range(numPlayers):
        players_name.append(input("Enter player {} name: ".format(player + 1)))
        players.append(drawCards(5))

    playerTurn = 0
    playDirection = 1
    playing = True
    discards.append(unoDeck.pop(0))
    splitCard = discards[0].split(" ", 1)
    currentColour = splitCard[0]
    if currentColour != "Wild":
        cardVal = splitCard[1]
    else:
        cardVal = "Any"

    while playing:
        showHand(playerTurn, players[playerTurn])
        print("Card on top of discard pile: {}".format(discards[-1]))
        if canPlay(currentColour, cardVal, players[playerTurn]):
            cardChosen = int(input("Which card do you want to play?"))
            while not canPlay(
                currentColour, cardVal, [players[playerTurn][cardChosen - 1]]
            ):
                cardChosen = int(
                    input("Not a valid card. Which card do you want to play?")
                )
            print("You played {}".format(players[playerTurn][cardChosen - 1]))
            discards.append(players[playerTurn].pop(cardChosen - 1))

            # cheak if player won
            if len(players[playerTurn]) == 0:
                playing = False
                # winner = "Player {}".format(playerTurn+1)
                winner = players_name[playerTurn]
            else:
                # cheak for special cards
                splitCard = discards[-1].split(" ", 1)
                currentColour = splitCard[0]
                if len(splitCard) == 1:
                    cardVal = "Any"
                else:
                    cardVal = splitCard[1]
                if currentColour == "Wild":
                    for x in range(len(colours)):
                        print("{}) {}".format(x + 1, colours[x]))
                    newColour = int(input("What colour would you like to choose? "))
                    while newColour < 1 or newColour > 4:
                        newColour = int(
                            input(
                                "Invalid option. What colour would you like to choose"
                            )
                        )
                    currentColour = colours[newColour - 1]
                if cardVal == "Reverse":
                    playDirection = playDirection * -1
                elif cardVal == "Skip":
                    playerTurn += playDirection
                    if playerTurn >= numPlayers:
                        playerTurn = 0
                    elif playerTurn < 0:
                        playerTurn = numPlayers - 1
                elif cardVal == "Draw Two":
                    playerDraw = playerTurn + playDirection
                    if playerDraw == numPlayers:
                        playerDraw = 0
                    elif playerDraw < 0:
                        playerDraw = numPlayers - 1
                    players[playerDraw].extend(drawCards(2))
                elif cardVal == "Draw Four":
                    playerDraw = playerTurn + playDirection
                    if playerDraw == numPlayers:
                        playerDraw = 0
                    elif playerDraw < 0:
                        playerDraw = numPlayers - 1
                    players[playerDraw].extend(drawCards(4))
                print("")
        else:
            print("You can't play. You have to draw a card.")
            players[playerTurn].extend(drawCards(1))

        playerTurn += playDirection
        if playerTurn >= numPlayers:
            playerTurn = 0
        elif playerTurn < 0:
            playerTurn = numPlayers - 1

    print("Game Over")
    print("{} is the Winner!".format(winner))


if __name__ == "__main__":
    main()
