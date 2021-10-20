import random
import time

SUITS = ('C', 'S', 'H', 'D')
RANKS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K')
VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 10, 'Q': 10, 'K': 10}


class card:
    def __init__(self, suit, rank):
        if (suit in SUITS) and (rank in RANKS):
            self.suit = suit
            self.rank = rank
        else:
            self.suit = None
            self.rank = None
            print("Invalid card: ", suit, rank)

    def __str__(self):
        return self.suit + self.rank

    def getRank(self):
        return self.rank

    def getSuit(self):
        return self.suit


class deck:
    def __init__(self):
        self.deck = [card(suit, rank) for suit in SUITS for rank in RANKS]

    def shuffle(self):
        random.shuffle(self.deck)

    def dealCard(self):
        return random.choice(self.deck)

    def __str__(self):
        print(self.deck)


# Begin play
# create two decks, one for each player.
print("Gathering brand new two decks of cards............\n")
deck1 = deck()
deck2 = deck()
time.sleep(5)
print('..........decks ready!!!\n')
print('Combining and shuffling both the decks..')
time.sleep(10)
# Shuffle the decks
deck1.shuffle()
deck2.shuffle()
# combine both the shuffled decks
combinedDeck = deck1.deck + deck2.deck
# ReShuffle the combined deck, cut it and distribute to two players.
random.shuffle(combinedDeck)
print("....decks have been combined and shuffled...\n")
print("------------------------------------------\n")
input("Enter a key to cut the deck..\n")
player1 = combinedDeck[0:52]
player2 = combinedDeck[52:]
print("Deck has been split into two and Human get a half and computer gets the other...\n")

# Begin play:
print("------------------------------------------\n")
print("player1 == Human\n")
print("player2 == Computer\n")
print("------------------------------------------\n")
print("player1 goes first...hit any key to place the card on the pile..\n")

centerPile = []
currentPlayer2Card = None

while len(player1) != 0 and len(player2) != 0:  # this needs a fix as it goes on an infinite loop on a success.
    switchPlayer = True
    while switchPlayer == True:
        for card in range(len(player1)):
            input("Enter any key to place a card!!!\n")
            currentPlayer1Card = player1[card].rank
            print("Your current card's rank: {}".format(currentPlayer1Card))
            centerPile.append(player1[card])
            player1.pop(card)
            switchPlayer = False
            if currentPlayer2Card == currentPlayer1Card:
                player1 = player1 + centerPile
                print("The human got a match and takes all the cards from center pile..")
            break
    while switchPlayer == False:
        for card in range(len(player2)):
            currentPlayer2Card = player2[card].rank
            print("Computer's current card's rank: {}".format(currentPlayer2Card))
            centerPile.append(player2[card])
            player2.pop(card)
            switchPlayer = True
            if currentPlayer1Card == currentPlayer2Card:
                player2 = player2 + centerPile
                print("Computer got a match and takes all the cards from center pile..")
            break

print("GAME OVER!!!\n")

print("Human has {} cards and computer has {}..".format(len(player1), len(player2)))
