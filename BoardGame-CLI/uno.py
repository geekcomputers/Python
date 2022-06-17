#      uno game      #

import random
"""
Generate the UNO deck of 108 cards.
Parameters: None
Return values: deck=>list
"""


def buildDeck():
    deck = []
    # example card:Red 7,Green 8, Blue skip
    colours = ["Red", "Green", "Yellow", "Blue"]
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Draw Two", "Skip", "Reverse"]
    wilds = ["Wild", "Wild Draw Four"]
    for colour in colours:
        for value in values:
            cardVal = f"{colour} {value}"
            deck.append(cardVal)
            if value != 0:
                deck.append(cardVal)
    for _ in range(4):
        deck.extend((wilds[0], wilds[1]))
    print(deck)
    return deck


"""
Shuffles a list of items passed into it
Parameters: deck=>list
Return values: deck=>list
"""


def shuffleDeck(deck):
    for cardPos in range(len(deck)):
        randPos = random.randint(0, 107)
        deck[cardPos], deck[randPos] = deck[randPos], deck[cardPos]
    return deck


"""Draw card function that draws a specified number of cards off the top of the deck
Parameters: numCards -> integer
Return: cardsDrawn -> list
"""


def drawCards(numCards):
    return [unoDeck.pop(0) for _ in range(numCards)]


"""
Print formatted list of player's hand
Parameter: player->integer , playerHand->list
Return: None
"""


def showHand(player, playerHand):
    print(f"Player {players_name[player]}'s Turn")
    print("Your Hand")
    print("------------------")
    for y, card in enumerate(playerHand, start=1):
        print(f"{y}) {card}")
    print("")


"""
Check whether a player is able to play a card, or not
Parameters: discardCard->string,value->string, playerHand->list
Return: boolean
"""


def canPlay(colour, value, playerHand):
    return any(
        "Wild" in card or colour in card or value in card
        for card in playerHand
    )


unoDeck = buildDeck()
unoDeck = shuffleDeck(unoDeck)
unoDeck = shuffleDeck(unoDeck)
players_name = []
players = []
colours = ["Red", "Green", "Yellow", "Blue"]
numPlayers = int(input("How many players?"))
while numPlayers < 2 or numPlayers > 4:
    numPlayers = int(
        input("Invalid. Please enter a number between 2-4.\nHow many players?"))
for player in range(numPlayers):
    players_name.append(input(f"Enter player {player + 1} name: "))
    players.append(drawCards(5))


playerTurn = 0
playDirection = 1
playing = True
discards = [unoDeck.pop(0)]
splitCard = discards[0].split(" ", 1)
currentColour = splitCard[0]
cardVal = splitCard[1] if currentColour != "Wild" else "Any"
while playing:
    showHand(playerTurn, players[playerTurn])
    print(f"Card on top of discard pile: {discards[-1]}")
    if canPlay(currentColour, cardVal, players[playerTurn]):
        cardChosen = int(input("Which card do you want to play?"))
        while not canPlay(currentColour, cardVal, [players[playerTurn][cardChosen-1]]):
            cardChosen = int(
                input("Not a valid card. Which card do you want to play?"))
        print(f"You played {players[playerTurn][cardChosen-1]}")
        discards.append(players[playerTurn].pop(cardChosen-1))

        # cheak if player won
        if len(players[playerTurn]) == 0:
            playing = False
            # winner = "Player {}".format(playerTurn+1)
            winner = players_name[playerTurn]
        else:
            # cheak for special cards
            splitCard = discards[-1].split(" ", 1)
            currentColour = splitCard[0]
            cardVal = "Any" if len(splitCard) == 1 else splitCard[1]
            if currentColour == "Wild":
                for x in range(len(colours)):
                    print(f"{x + 1}) {colours[x]}")
                newColour = int(
                    input("What colour would you like to choose? "))
                while newColour < 1 or newColour > 4:
                    newColour = int(
                        input("Invalid option. What colour would you like to choose"))
                currentColour = colours[newColour-1]
            if cardVal == "Reverse":
                playDirection = playDirection * -1
            elif cardVal == "Skip":
                playerTurn += playDirection
                if playerTurn >= numPlayers:
                    playerTurn = 0
                elif playerTurn < 0:
                    playerTurn = numPlayers-1
            elif cardVal == "Draw Two":
                playerDraw = playerTurn+playDirection
                if playerDraw == numPlayers:
                    playerDraw = 0
                elif playerDraw < 0:
                    playerDraw = numPlayers-1
                players[playerDraw].extend(drawCards(2))
            elif cardVal == "Draw Four":
                playerDraw = playerTurn+playDirection
                if playerDraw == numPlayers:
                    playerDraw = 0
                elif playerDraw < 0:
                    playerDraw = numPlayers-1
                players[playerDraw].extend(drawCards(4))
            print("")
    else:
        print("You can't play. You have to draw a card.")
        players[playerTurn].extend(drawCards(1))

    playerTurn += playDirection
    if playerTurn >= numPlayers:
        playerTurn = 0
    elif playerTurn < 0:
        playerTurn = numPlayers-1

print("Game Over")
print(f"{winner} is the Winner!")
