import random
import time

class Card:
    """Represents a single playing card"""
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        
    def get_value(self, hand_value: int) -> int:
        """Returns the value of the card in the context of the current hand"""
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            # Ace can be 11 or 1, choose the optimal value
            return 11 if hand_value + 11 <= 21 else 1
        else:
            return int(self.rank)
    
    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

class Deck:
    """Represents a deck of playing cards"""
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        """Reset the deck to its initial state"""
        suits = ['♥', '♦', '♣', '♠']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(rank, suit) for suit in suits for rank in ranks] * 4
        self.shuffle()
    
    def shuffle(self) -> None:
        """Shuffle the deck of cards"""
        random.shuffle(self.cards)
    
    def deal_card(self) -> Card:
        """Deal a single card from the deck"""
        if not self.cards:
            self.reset()  # Auto-reset the deck if empty
        return self.cards.pop()

class Hand:
    """Represents a player's or dealer's hand of cards"""
    def __init__(self):
        self.cards = []
    
    def add_card(self, card: Card) -> None:
        """Add a card to the hand"""
        self.cards.append(card)
    
    def get_value(self) -> int:
        """Calculate the value of the hand, considering soft and hard aces"""
        total = 0
        aces = 0
        
        for card in self.cards:
            if card.rank == 'A':
                aces += 1
            total += card.get_value(total)
        
        # Adjust for aces (if total > 21, convert aces from 11 to 1)
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
            
        return total
    
    def is_bust(self) -> bool:
        """Check if the hand is bust (over 21)"""
        return self.get_value() > 21
    
    def is_blackjack(self) -> bool:
        """Check if the hand is a blackjack (21 with 2 cards)"""
        return len(self.cards) == 2 and self.get_value() == 21
    
    def __str__(self) -> str:
        return ", ".join(str(card) for card in self.cards)

class BlackjackGame:
    """Main game class that manages the game flow"""
    def __init__(self):
        self.deck = Deck()
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.player_balance = 1000  # Starting balance
        self.current_bet = 0
        
    def display_welcome_message(self) -> None:
        """Display the welcome message and game introduction"""
        title = f"""
{"*"*58}
    Welcome to the Casino - BLACK JACK !
{"*"*58}
"""
        self._animate_text(title, 0.01)
        time.sleep(1)
        
        messages = [
            "So finally you are here to test your luck...",
            "I mean your fortune!",
            "Let's see how lucky you are. Wish you all the best!",
            "Loading...",
            "Still loading...",
            "So you're still here. I gave you a chance to leave, but no problem.",
            "Maybe you trust your fortune a lot. Let's begin then!"
        ]
        
        for message in messages:
            self._animate_text(message)
            time.sleep(0.8)
    
    def _animate_text(self, text: str, delay: float = 0.03) -> None:
        """Animate text by printing each character with a small delay"""
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()  # Newline at the end
    
    def place_bet(self) -> None:
        """Prompt the player to place a bet"""
        while True:
            try:
                print(f"\nYour current balance: ${self.player_balance}")
                bet = int(input("Place your bet: $"))
                if 1 <= bet <= self.player_balance:
                    self.current_bet = bet
                    self.player_balance -= bet
                    break
                else:
                    print(f"Please bet between $1 and ${self.player_balance}")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    
    def deal_initial_cards(self) -> None:
        """Deal the initial cards to the player and dealer"""
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        
        # Deal two cards to the player and dealer
        for _ in range(2):
            self.player_hand.add_card(self.deck.deal_card())
            self.dealer_hand.add_card(self.deck.deal_card())
        
        # Show the player's hand and the dealer's up card
        self._display_hands(show_dealer=False)
    
    def _display_hands(self, show_dealer: bool = True) -> None:
        """Display the player's and dealer's hands"""
        print("\n" + "-"*40)
        if show_dealer:
            print(f"Dealer's Hand ({self.dealer_hand.get_value()}): {self.dealer_hand}")
        else:
            print(f"Dealer's Hand: X, {self.dealer_hand.cards[1]}")
        print(f"Your Hand ({self.player_hand.get_value()}): {self.player_hand}")
        print("-"*40 + "\n")
    
    def player_turn(self) -> None:
        """Handle the player's turn (hit or stand)"""
        while not self.player_hand.is_bust() and not self.player_hand.is_blackjack():
            choice = input("Do you want to [H]it or [S]tand? ").strip().lower()
            
            if choice == 'h':
                self.player_hand.add_card(self.deck.deal_card())
                self._display_hands(show_dealer=False)
                
                if self.player_hand.is_bust():
                    print("BUST! You went over 21.")
                    return
                elif self.player_hand.is_blackjack():
                    print("BLACKJACK! You got 21!")
                    return
            elif choice == 's':
                print("You chose to stand.")
                return
            else:
                print("Invalid choice. Please enter 'H' or 'S'.")
    
    def dealer_turn(self) -> None:
        """Handle the dealer's turn (automatically hits until 17 or higher)"""
        print("\nDealer's Turn...")
        self._display_hands(show_dealer=True)
        
        while self.dealer_hand.get_value() < 17:
            print("Dealer hits...")
            self.dealer_hand.add_card(self.deck.deal_card())
            time.sleep(1)
            self._display_hands(show_dealer=True)
            
            if self.dealer_hand.is_bust():
                print("Dealer BUSTS!")
                return
    
    def determine_winner(self) -> None:
        """Determine the winner of the game and update the player's balance"""
        player_value = self.player_hand.get_value()
        dealer_value = self.dealer_hand.get_value()
        
        if self.player_hand.is_bust():
            print(f"{'*'*20} Dealer Wins! {'*'*20}")
            return
        elif self.dealer_hand.is_bust():
            winnings = self.current_bet * 2
            self.player_balance += winnings
            print(f"{'*'*20} You Win! +${winnings} {'*'*20}")
            return
        elif self.player_hand.is_blackjack() and not self.dealer_hand.is_blackjack():
            # Blackjack pays 3:2
            winnings = int(self.current_bet * 2.5)
            self.player_balance += winnings
            print(f"{'*'*15} Blackjack! You Win! +${winnings} {'*'*15}")
            return
        elif self.dealer_hand.is_blackjack() and not self.player_hand.is_blackjack():
            print(f"{'*'*20} Dealer Blackjack! Dealer Wins! {'*'*20}")
            return
        
        # Compare values if neither is bust or blackjack
        if player_value > dealer_value:
            winnings = self.current_bet * 2
            self.player_balance += winnings
            print(f"{'*'*20} You Win! +${winnings} {'*'*20}")
        elif player_value < dealer_value:
            print(f"{'*'*20} Dealer Wins! {'*'*20}")
        else:
            # Tie (push)
            self.player_balance += self.current_bet
            print(f"{'*'*20} It's a Tie! Your bet is returned. {'*'*20}")
    
    def play_again(self) -> bool:
        """Ask the player if they want to play another round"""
        if self.player_balance <= 0:
            print("\nYou're out of money! Game over.")
            return False
            
        choice = input("\nDo you want to play another round? [Y/N] ").strip().lower()
        return choice == 'y'
    
    def run(self) -> None:
        """Run the main game loop"""
        self.display_welcome_message()
        
        while True:
            if self.player_balance <= 0:
                print("\nYou've run out of money! Thanks for playing.")
                break
                
            self.place_bet()
            self.deal_initial_cards()
            
            # Check for immediate blackjack
            if self.player_hand.is_blackjack():
                self._display_hands(show_dealer=True)
                self.determine_winner()
            else:
                self.player_turn()
                if not self.player_hand.is_bust():
                    self.dealer_turn()
                    self.determine_winner()
            
            if not self.play_again():
                print(f"\nThanks for playing! Your final balance: ${self.player_balance}")
                break

if __name__ == "__main__":
    game = BlackjackGame()
    game.run()