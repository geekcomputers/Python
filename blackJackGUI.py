
from __future__ import print_function
import random
import simplegui
CARD_SIZE = (72, 96)
CARD_CENTER = (36, 48)
card_images = simplegui.load_image("http://storage.googleapis.com/codeskulptor-assets/cards_jfitz.png")

in_play = False
outcome = ""
score = 0

SUITS = ('C', 'S', 'H', 'D')
RANKS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K')
VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 10, 'Q': 10, 'K': 10}


class Card:
    def __init__(self, suit, rank):
        if (suit in SUITS) and (rank in RANKS):
            self.suit = suit
            self.rank = rank
        else:
            self.suit = None
            self.rank = None
            print(("Invalid card: ", suit, rank))

    def __str__(self):
        return self.suit + self.rank

    def get_suit(self):
        return self.suit

    def get_rank(self):
        return self.rank

    def draw(self, canvas, pos):
        card_loc = (CARD_CENTER[0] + CARD_SIZE[0] * RANKS.index(self.rank),
                    CARD_CENTER[1] + CARD_SIZE[1] * SUITS.index(self.suit))
        canvas.draw_image(card_images, card_loc, CARD_SIZE, [pos[0] + CARD_CENTER[0], pos[1] + CARD_CENTER[1]],
                          CARD_SIZE)


def string_list_join(string, string_list):
    ans = string + " contains "
    for i in range(len(string_list)):
        ans += str(string_list[i]) + " "
    return ans


class Hand:
    def __init__(self):
        self.hand = []

    def __str__(self):
        return string_list_join("Hand", self.hand)

    def add_card(self, card):
        self.hand.append(card)

    def get_value(self):
        var = []
        self.hand_value = 0
        for card in self.hand:
            card = str(card)
            if card[1] in VALUES:
                self.hand_value += VALUES[card[1]]
                var.append(card[1])
        if 'A' not in var:
            return self.hand_value
        if self.hand_value + 10 <= 21:
            return self.hand_value + 10
        else:
            return self.hand_value

    def draw(self, canvas, pos):
        for card in self.hand:
            card = str(card)
            Card(card[0], card[1]).draw(canvas, pos)
            pos[0] += 36


class Deck:
    def __init__(self):
        self.Deck = [Card(suit, rank) for suit in SUITS for rank in RANKS]

    def shuffle(self):
        random.shuffle(self.Deck)

    def deal_card(self):
        return random.choice(self.Deck)

    def __str__(self):
        return string_list_join("Deck", self.Deck)


def deal():
    global outcome, in_play, score1, score2, player_card, dealer_card, deck
    outcome = ""
    player_card = Hand()
    dealer_card = Hand()
    deck = Deck()
    for i in range(2):
        player_card.add_card(deck.deal_card())
        dealer_card.add_card(deck.deal_card())

    in_play = True
    score1 = str(player_card.get_value())
    score2 = str(dealer_card.get_value())


def stand():
    if in_play == True:
        while dealer_card.get_value() < 17:
            dealer_card.add_card(deck.deal_card())
    if dealer_card.get_value() > 21:
        outcome = "you won!!"
    elif player_card.get_value() <= dealer_card.get_value():
        outcome = "you lose"
    else:
        outcome = "you won!!"
    score1 = str(player_card.get_value())
    score2 = str(dealer_card.get_value())


def hit():
    global outcome, in_play, score1, score2, player_card, dealer_card, deck
    if in_play == True:
        player_card.add_card(deck.deal_card())

    if player_card.get_value() > 21:
        outcome = "you are busted"
        in_play = False

    score1 = str(player_card.get_value())
    score2 = str(dealer_card.get_value())


def draw(canvas):
    canvas.draw_text(outcome, [250, 150], 25, 'White')
    canvas.draw_text("BlackJack", [250, 50], 40, 'Black')
    canvas.draw_text(score1, [100, 100], 40, 'Red')

    player_card.draw(canvas, [20, 300])
    dealer_card.draw(canvas, [300, 300])
    canvas.draw_text(score2, [400, 100], 40, 'Red')


frame = simplegui.create_frame("Blackjack", 600, 600)
frame.set_canvas_background("Green")

frame.add_button("Deal", deal, 200)
frame.add_button("Hit", hit, 200)
frame.add_button("Stand", stand, 200)
frame.set_draw_handler(draw)

deal()
frame.start()
