import random


# ANSI color codes for console output
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colorize_card(card: str) -> str:
    """
    Colorize the card text based on its color for better console visibility.
    
    Args:
        card (str): The card description (e.g., "Red 5").
    
    Returns:
        str: The colorized card text using ANSI escape codes.
    """
    if card.startswith("Red"):
        return f"{Colors.RED}{card}{Colors.RESET}"
    elif card.startswith("Green"):
        return f"{Colors.GREEN}{card}{Colors.RESET}"
    elif card.startswith("Yellow"):
        return f"{Colors.YELLOW}{card}{Colors.RESET}"
    elif card.startswith("Blue"):
        return f"{Colors.BLUE}{card}{Colors.RESET}"
    elif "Wild" in card:
        return f"{Colors.PURPLE}{card}{Colors.RESET}"
    return card

def build_deck() -> list[str]:
    """
    Generate a standard UNO deck consisting of 108 cards.
    
    Returns:
        List[str]: A list containing all UNO cards as strings.
    """
    deck: list[str] = []
    colors: list[str] = ["Red", "Green", "Yellow", "Blue"]
    values: list[int | str] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Draw Two", "Skip", "Reverse"]
    wilds: list[str] = ["Wild", "Wild Draw Four"]
    
    # Add numbered and action cards
    for color in colors:
        for value in values:
            card = f"{color} {value}"
            deck.append(card)
            if value != 0:  # Each non-zero card appears twice
                deck.append(card)
    
    # Add wild cards
    for _ in range(4):
        deck.append(wilds[0])
        deck.append(wilds[1])
    
    print(f"Deck built with {len(deck)} cards.")
    return deck

def shuffle_deck(deck: list[str]) -> list[str]:
    """
    Shuffle the given deck using the Fisher-Yates algorithm for a uniform random permutation.
    
    Args:
        deck (List[str]): The deck of cards to shuffle.
    
    Returns:
        List[str]: The shuffled deck.
    """
    for i in range(len(deck) - 1, 0, -1):
        j = random.randint(0, i)
        deck[i], deck[j] = deck[j], deck[i]
    print("Deck shuffled.")
    return deck

def draw_cards(num_cards: int, deck: list[str], discards: list[str]) -> list[str]:
    """
    Draw a specified number of cards from the deck. 
    Reshuffles the discard pile into the deck if it's empty (except the top card).
    
    Args:
        num_cards (int): Number of cards to draw.
        deck (List[str]): The main deck to draw from.
        discards (List[str]): The discard pile.
    
    Returns:
        List[str]: The cards drawn from the deck.
    """
    drawn_cards: list[str] = []
    for _ in range(num_cards):
        if not deck:  # Reshuffle discard pile if deck is empty
            print(f"{Colors.BOLD}Reshuffling discard pile into deck...{Colors.RESET}")
            deck = shuffle_deck(discards[:-1])  # Keep the top discard card
            discards.clear()
            discards.append(deck.pop())  # Move top card to discard pile
        
        drawn_cards.append(deck.pop(0))
    
    return drawn_cards

def show_hand(player_name: str, player_hand: list[str]) -> None:
    """
    Display the player's current hand in a formatted and colorized manner.
    
    Args:
        player_name (str): The name of the player.
        player_hand (List[str]): The player's current hand of cards.
    """
    print(f"\n{Colors.BOLD}=== {player_name}'s Turn ==={Colors.RESET}")
    print(f"Your Hand ({len(player_hand)} cards):")
    print("--------------------------------")
    for i, card in enumerate(player_hand, 1):
        print(f"{i}) {colorize_card(card)}")
    print("")

def can_play(current_color: str, current_value: str, player_hand: list[str]) -> bool:
    """
    Check if the player can play any card from their hand based on the current discard pile.
    
    Args:
        current_color (str): The current active color.
        current_value (str): The current active value.
        player_hand (List[str]): The player's current hand.
    
    Returns:
        bool: True if the player can play at least one card, False otherwise.
    """
    for card in player_hand:
        if "Wild" in card:
            return True
        card_color, card_value = card.split(" ", 1)
        if card_color == current_color or card_value == current_value:
            return True
    return False

def get_valid_input(prompt: str, min_val: int, max_val: int, input_type: type = int) -> int | str:
    """
    Get valid user input within a specified range and type.
    
    Args:
        prompt (str): The message to display.
        min_val (int): Minimum acceptable value (inclusive).
        max_val (int): Maximum acceptable value (inclusive).
        input_type (type): Expected data type (int or str).
    
    Returns:
        Union[int, str]: Validated user input.
    """
    while True:
        user_input = input(prompt)
        
        try:
            if input_type == int:
                value = int(user_input)
                if min_val <= value <= max_val:
                    return value
                print(f"Please enter a number between {min_val} and {max_val}.")
            elif input_type == str:
                if user_input.lower() in ['y', 'n']:
                    return user_input.lower()
                print("Please enter 'y' or 'n'.")
            else:
                print(f"Unsupported input type: {input_type}")
                return None
                
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")

def show_game_status(players_name: list[str], players: list[list[str]], play_direction: int, player_turn: int, num_players: int) -> None:
    """
    Display the current game status including player hands, direction, and next player.
    
    Args:
        players_name (List[str]): List of player names.
        players (List[List[str]]): List of each player's hand.
        play_direction (int): 1 for clockwise, -1 for counter-clockwise.
        player_turn (int): Index of the current player.
        num_players (int): Total number of players.
    """
    print(f"\n{Colors.BOLD}=== Game Status ==={Colors.RESET}")
    for i, name in enumerate(players_name):
        print(f"{name}: {len(players[i])} cards")
    direction = "Clockwise" if play_direction == 1 else "Counter-clockwise"
    print(f"Direction: {Colors.BOLD}{direction}{Colors.RESET}")
    next_player = (player_turn + play_direction) % num_players
    print(f"Next player: {Colors.BOLD}{players_name[next_player]}{Colors.RESET}")
    print("-------------------")

def main() -> None:
    """
    Main function to initialize and run the UNO game.
    """
    print(f"{Colors.BOLD}{Colors.UNDERLINE}Welcome to UNO!{Colors.RESET}")
    
    # Initialize game components
    uno_deck = build_deck()
    uno_deck = shuffle_deck(uno_deck)
    discards: list[str] = []

    players_name: list[str] = []
    players: list[list[str]] = []
    colors: list[str] = ["Red", "Green", "Yellow", "Blue"]

    # Get number of players
    num_players = get_valid_input("How many players? (2-4): ", 2, 4)

    # Get player names
    for i in range(num_players):
        while True:
            name = input(f"Enter player {i+1} name: ").strip()
            if name:
                players_name.append(name)
                break
            print("Name cannot be empty. Please try again.")

    print(f"\n{Colors.BOLD}=== Game Starting ==={Colors.RESET}")
    print(f"Players: {', '.join(players_name)}")

    # Deal initial cards
    for i in range(num_players):
        players.append(draw_cards(7, uno_deck, discards))
        print(f"{players_name[i]} received 7 cards.")

    # Initialize game state
    player_turn: int = 0
    play_direction: int = 1  # 1 for clockwise, -1 for counter-clockwise
    game_active: bool = True

    # Start with first card on discard pile
    discards.append(uno_deck.pop(0))
    top_card = discards[-1].split(" ", 1)
    current_color: str = top_card[0]
    current_value: str = top_card[1] if len(top_card) > 1 else "Any"

    # Handle wild cards as starting card
    if current_color == "Wild":
        print("Starting card is Wild. Choosing random color...")
        current_color = random.choice(colors)

    print(f"\nGame begins with: {colorize_card(discards[-1])} ({current_color})")

    # Main game loop
    while game_active:
        current_hand = players[player_turn]
        
        # Show game status before each turn
        show_game_status(players_name, players, play_direction, player_turn, num_players)
        show_hand(players_name[player_turn], current_hand)
        print(f"Current card: {colorize_card(discards[-1])} ({current_color})")
        
        # Check if player can play
        if can_play(current_color, current_value, current_hand):
            valid_moves = [i+1 for i, card in enumerate(current_hand) 
                          if 'Wild' in card or card.startswith(current_color) or current_value in card]
            print(f"Valid moves: {[colorize_card(current_hand[i-1]) for i in valid_moves]}")
            
            # Get valid card choice
            card_count = len(current_hand)
            card_chosen = get_valid_input(
                f"Which card do you want to play? (1-{card_count}): ", 1, card_count
            )
            
            # Validate selected card
            while True:
                selected_card = current_hand[card_chosen - 1]
                card_color, card_value = selected_card.split(" ", 1) if "Wild" not in selected_card else (selected_card, "")
                
                if "Wild" in selected_card or card_color == current_color or card_value == current_value:
                    break
                else:
                    print("Invalid card selection.")
                    card_chosen = get_valid_input(
                        f"Please choose a valid card. (1-{card_count}): ", 1, card_count
                    )
            
            # Play the card
            played_card = current_hand.pop(card_chosen - 1)
            discards.append(played_card)
            print(f"\n{Colors.BOLD}{players_name[player_turn]} played: {colorize_card(played_card)}{Colors.RESET}")
            
            # Check if player won
            if not current_hand:
                game_active = False
                winner = players_name[player_turn]
                print(f"\n{Colors.BOLD}{Colors.UNDERLINE}=== Game Over! ==={Colors.RESET}")
                print(f"{Colors.BOLD}{winner} is the Winner!{Colors.RESET}")
                break
            
            # Process special cards
            top_card = discards[-1].split(" ", 1)
            current_color = top_card[0]
            current_value = top_card[1] if len(top_card) > 1 else "Any"
            
            # Handle wild cards
            if current_color == "Wild":
                print(f"\n{Colors.BOLD}=== Wild Card ==={Colors.RESET}")
                print("Choose a new color:")
                for i, color in enumerate(colors, 1):
                    print(f"{i}) {color}")
                color_choice = get_valid_input("Enter color choice (1-4): ", 1, 4)
                current_color = colors[color_choice - 1]
                print(f"Color changed to {current_color}")
            
            # Handle action cards
            if current_value == "Reverse":
                play_direction *= -1
                print(f"\n{Colors.BOLD}=== Reverse ==={Colors.RESET}")
                print("Direction reversed!")
            elif current_value == "Skip":
                skipped_player = (player_turn + play_direction) % num_players
                player_turn = skipped_player
                print(f"\n{Colors.BOLD}=== Skip ==={Colors.RESET}")
                print(f"{players_name[skipped_player]} has been skipped!")
            elif current_value == "Draw Two":
                next_player = (player_turn + play_direction) % num_players
                drawn_cards = draw_cards(2, uno_deck, discards)
                players[next_player].extend(drawn_cards)
                print(f"\n{Colors.BOLD}=== Draw Two ==={Colors.RESET}")
                print(f"{players_name[next_player]} draws 2 cards!")
            elif current_value == "Draw Four":
                next_player = (player_turn + play_direction) % num_players
                drawn_cards = draw_cards(4, uno_deck, discards)
                players[next_player].extend(drawn_cards)
                print(f"\n{Colors.BOLD}=== Draw Four ==={Colors.RESET}")
                print(f"{players_name[next_player]} draws 4 cards!")
        else:
            print(f"\n{Colors.BOLD}You can't play. Drawing a card...{Colors.RESET}")
            drawn_card = draw_cards(1, uno_deck, discards)[0]
            players[player_turn].append(drawn_card)
            print(f"You drew: {colorize_card(drawn_card)}")
            
            # Check if drawn card can be played
            card_color, card_value = drawn_card.split(" ", 1) if "Wild" not in drawn_card else (drawn_card, "")
            if "Wild" in drawn_card or card_color == current_color or card_value == current_value:
                print("You can play the drawn card!")
                play_choice = get_valid_input("Do you want to play it? (y/n): ", 0, 1, str)
                if play_choice == 'y':
                    players[player_turn].remove(drawn_card)
                    discards.append(drawn_card)
                    print(f"\n{Colors.BOLD}{players_name[player_turn]} played: {colorize_card(drawn_card)}{Colors.RESET}")
                    
                    # Process special cards
                    top_card = discards[-1].split(" ", 1)
                    current_color = top_card[0]
                    current_value = top_card[1] if len(top_card) > 1 else "Any"
                    
                    if current_color == "Wild":
                        print(f"\n{Colors.BOLD}=== Wild Card ==={Colors.RESET}")
                        print("Choose a new color:")
                        for i, color in enumerate(colors, 1):
                            print(f"{i}) {color}")
                        color_choice = get_valid_input("Enter color choice (1-4): ", 1, 4)
                        current_color = colors[color_choice - 1]
                        print(f"Color changed to {current_color}")
                    
                    if current_value == "Reverse":
                        play_direction *= -1
                        print(f"\n{Colors.BOLD}=== Reverse ==={Colors.RESET}")
                        print("Direction reversed!")
                    elif current_value == "Skip":
                        skipped_player = int((player_turn + play_direction) % num_players)  # Ensure integer
                        player_turn = skipped_player
                        print(f"\n{Colors.BOLD}=== Skip ==={Colors.RESET}")
                        print(f"{players_name[skipped_player]} has been skipped!")
                    elif current_value == "Draw Two":
                        next_player = int((player_turn + play_direction) % num_players)  # Ensure integer
                        drawn_cards = draw_cards(2, uno_deck, discards)
                        players[next_player].extend(drawn_cards)
                        print(f"\n{Colors.BOLD}=== Draw Two ==={Colors.RESET}")
                        print(f"{players_name[next_player]} draws 2 cards!")
                    elif current_value == "Draw Four":
                        next_player = int((player_turn + play_direction) % num_players)  # Ensure integer
                        drawn_cards = draw_cards(4, uno_deck, discards)
                        players[next_player].extend(drawn_cards)
                        print(f"\n{Colors.BOLD}=== Draw Four ==={Colors.RESET}")
                        print(f"{players_name[next_player]} draws 4 cards!")
        
        # Move to next player
        player_turn = int((player_turn + play_direction) % num_players)  # Ensure integer

if __name__ == "__main__":
    main()