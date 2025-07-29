import random


class Colour:
    BLACK: str = "\033[30m"
    RED: str = "\033[91m"
    GREEN: str = "\033[32m"
    END: str = "\033[0m"


suits: tuple[str, ...] = (
    f"{Colour.RED}Hearts{Colour.END}",
    f"{Colour.RED}Diamonds{Colour.END}",
    f"{Colour.BLACK}Spades{Colour.END}",
    f"{Colour.BLACK}Clubs{Colour.END}",
)

ranks: tuple[str, ...] = (
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Jack",
    "Queen",
    "King",
    "Ace",
)

values: dict[str, int] = {
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
    "Six": 6,
    "Seven": 7,
    "Eight": 8,
    "Nine": 9,
    "Ten": 10,
    "Jack": 10,
    "Queen": 10,
    "King": 10,
    "Ace": 11,
}

playing: bool = True


class Card:
    def __init__(self, suit: str, rank: str) -> None:
        self.suit: str = suit
        self.rank: str = rank

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"


class Deck:
    def __init__(self) -> None:
        self.deck: list[Card] = []
        for suit in suits:
            for rank in ranks:
                self.deck.append(Card(suit, rank))

    def __str__(self) -> str:
        deck_comp: str = ""
        for card in self.deck:
            deck_comp += f"\n {card}"
        return deck_comp

    def shuffle(self) -> None:
        random.shuffle(self.deck)

    def deal(self) -> Card:
        return self.deck.pop()


class Hand:
    def __init__(self) -> None:
        self.cards: list[Card] = []
        self.value: int = 0
        self.aces: int = 0  # To track aces

    def add_card(self, card: Card) -> None:
        self.cards.append(card)
        self.value += values[card.rank]
        if card.rank == "Ace":
            self.aces += 1

    def adjust_for_ace(self) -> None:
        while self.value > 21 and self.aces > 0:
            self.value -= 10
            self.aces -= 1


class Chips:
    def __init__(self, total: int = 100) -> None:
        self.total: int = total
        self.bet: int = 0

    def win_bet(self) -> None:
        self.total += self.bet

    def lose_bet(self) -> None:
        self.total -= self.bet


def take_bet(chips: Chips) -> None:
    while True:
        try:
            chips.bet = int(input("How many chips would you like to bet? "))
        except ValueError:
            print("Your bet must be an integer! Please try again.")
        else:
            if chips.bet > chips.total or chips.bet <= 0:
                print(
                    "Invalid bet! Must be positive and not exceed your balance. "
                    f"Current balance: {chips.total}"
                )
            else:
                break


def hit(deck: Deck, hand: Hand) -> None:
    hand.add_card(deck.deal())
    hand.adjust_for_ace()


def hit_or_stand(deck: Deck, hand: Hand) -> None:
    global playing  # pylint: disable=global-statement

    while True:
        choice: str = input("Would you like to Hit or Stand? Enter '1' or '0': ")

        if choice.strip().lower() == "1":
            hit(deck, hand)
        elif choice.strip().lower() == "0":
            print("You chose to stand. Dealer will play.")
            playing = False
        else:
            print("Invalid input. Please enter '1' (Hit) or '0' (Stand).")
            continue
        break


def show_some(player: Hand, dealer: Hand) -> None:
    print("\nDealer's Hand:")
    print(" { hidden card }")
    print(f" {dealer.cards[1]}")
    print("\nYour Hand:")
    for card in player.cards:
        print(f" {card}")


def show_all(player: Hand, dealer: Hand) -> None:
    print("\nDealer's Hand:")
    for card in dealer.cards:
        print(f" {card}")
    print(f"Dealer's Total: {dealer.value}")

    print("\nYour Hand:")
    for card in player.cards:
        print(f" {card}")
    print(f"Your Total: {player.value}")


def player_busts(player: Hand, dealer: Hand, chips: Chips) -> None:
    print("You bust! Dealer wins this round.")
    chips.lose_bet()


def player_wins(player: Hand, dealer: Hand, chips: Chips) -> None:
    print("Congratulations! You win this round!")
    chips.win_bet()


def dealer_busts(player: Hand, dealer: Hand, chips: Chips) -> None:
    print("Dealer busts! You win this round!")
    chips.win_bet()


def dealer_wins(player: Hand, dealer: Hand, chips: Chips) -> None:
    print("Dealer wins this round!")
    chips.lose_bet()


def push(player: Hand, dealer: Hand) -> None:
    print("It's a tie! Push.")


def main() -> None:
    player_chips: Chips = Chips()

    while True:
        print(
            "\t**********************************************************\n"
            "\t                     Casino Blackjack\n"
            "\t**********************************************************"
        )
        print(
            f"{Colour.BLACK}\t                                   ***************\n"
            "\t                                   * A           *\n"
            "\t                                   *             *\n"
            "\t                                   *      *      *\n"
            "\t                                   *     ***     *\n"
            "\t                                   *    *****    *\n"
            "\t                                   *     ***     *\n"
            "\t                                   *      *      *\n"
            "\t                                   *             *\n"
            "\t                                   *             *\n"
            f"\t                                   ***************{Colour.END}"
        )
        print(
            "\nRules: Get as close to 21 as possible without exceeding it.\n"
            "Aces count as 1 or 11."
        )

        deck: Deck = Deck()
        deck.shuffle()

        player_hand: Hand = Hand()
        player_hand.add_card(deck.deal())
        player_hand.add_card(deck.deal())

        dealer_hand: Hand = Hand()
        dealer_hand.add_card(deck.deal())
        dealer_hand.add_card(deck.deal())

        take_bet(player_chips)
        show_some(player_hand, dealer_hand)

        global playing  # pylint: disable=global-statement
        playing = True

        while playing:
            hit_or_stand(deck, player_hand)
            show_some(player_hand, dealer_hand)

            if player_hand.value > 21:
                player_busts(player_hand, dealer_hand, player_chips)
                break

        if player_hand.value <= 21:
            while dealer_hand.value < 17:
                hit(deck, dealer_hand)

            show_all(player_hand, dealer_hand)

            if dealer_hand.value > 21:
                dealer_busts(player_hand, dealer_hand, player_chips)
            elif dealer_hand.value > player_hand.value:
                dealer_wins(player_hand, dealer_hand, player_chips)
            elif dealer_hand.value < player_hand.value:
                player_wins(player_hand, dealer_hand, player_chips)
            else:
                push(player_hand, dealer_hand)

        print(f"\nYour current balance: {player_chips.total}")

        if player_chips.total > 0:
            new_game: str = input("Play another hand? Enter '1' (yes) or '0' (no): ")
            if new_game.strip().lower() == "1":
                continue
            print(
                f"Thanks for playing!\n{Colour.GREEN}"
                "\t$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                f"\t      Congratulations! You won {player_chips.total} coins!\n"
                "\t$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
                f"{Colour.END}"
            )
            break
        else:
            print(
                "Oops! You've run out of chips.\n"
                "Thanks for playing! Come back soon to Casino Blackjack!"
            )
            break


if __name__ == "__main__":
    main()
