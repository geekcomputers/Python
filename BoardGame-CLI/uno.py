import random

def build_deck() -> list:
    """
    Generate the UNO deck of 108 cards.
    
    Returns:
        list: A list containing all 108 UNO cards.
    """
    deck = []
    colors = ["Red", "Green", "Yellow", "Blue"]
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Draw Two", "Skip", "Reverse"]
    wilds = ["Wild", "Wild Draw Four"]
    
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

def shuffle_deck(deck: list) -> list:
    """
    Shuffle the given deck of cards using Fisher-Yates algorithm.
    
    Args:
        deck (list): The deck of cards to shuffle.
    
    Returns:
        list: The shuffled deck.
    """
    for i in range(len(deck) - 1, 0, -1):
        j = random.randint(0, i)
        deck[i], deck[j] = deck[j], deck[i]
    print("Deck shuffled.")
    return deck

def draw_cards(num_cards: int, deck: list, discards: list) -> list:
    """
    Draw a specified number of cards from the top of the deck.
    Reshuffles discard pile if deck is empty.
    
    Args:
        num_cards (int): Number of cards to draw.
        deck (list): The deck to draw from.
        discards (list): The discard pile for reshuffling.
    
    Returns:
        list: The cards drawn from the deck.
    """
    drawn_cards = []
    for _ in range(num_cards):
        if not deck:  # If deck is empty, reshuffle discard pile
            print("Reshuffling discard pile into deck...")
            deck = shuffle_deck(discards[:-1])  # Keep the top discard card
            discards.clear()
            discards.append(deck.pop())  # Move top card to discard pile
        
        drawn_cards.append(deck.pop(0))
    
    return drawn_cards

def show_hand(player: int, player_hand: list) -> None:
    """
    Display the player's current hand in a formatted manner.
    
    Args:
        player (int): The player number.
        player_hand (list): The player's current hand of cards.
    """
    print(f"\n=== {players_name[player]}'s Turn ===")
    print(f"Your Hand ({len(player_hand)} cards):")
    print("--------------------------------")
    for i, card in enumerate(player_hand, 1):
        print(f"{i}) {card}")
    print("")

def can_play(current_color: str, current_value: str, player_hand: list) -> bool:
    """
    Check if the player can play any card from their hand.
    
    Args:
        current_color (str): The current color on the discard pile.
        current_value (str): The current value on the discard pile.
        player_hand (list): The player's current hand of cards.
    
    Returns:
        bool: True if the player can play a card, False otherwise.
    """
    for card in player_hand:
        if "Wild" in card:
            return True
        card_color, card_value = card.split(" ", 1)
        if card_color == current_color or card_value == current_value:
            return True
    return False

def get_valid_input(prompt: str, min_val: int, max_val: int, input_type: type = int) -> any:
    """
    Get a valid input from the user within a specified range and type.
    
    Args:
        prompt (str): The message to display to the user.
        min_val (int): The minimum acceptable value (for numeric inputs).
        max_val (int): The maximum acceptable value (for numeric inputs).
        input_type (type): The expected data type of the input.
    
    Returns:
        any: A valid input of the specified type.
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

def show_game_status() -> None:
    """Display current game status including player hands and scores"""
    print("\n=== Game Status ===")
    for i, name in enumerate(players_name):
        print(f"{name}: {len(players[i])} cards")
    print(f"Direction: {'Clockwise' if play_direction == 1 else 'Counter-clockwise'}")
    print(f"Next player: {players_name[(player_turn + play_direction) % num_players]}")
    print("-------------------")

# Initialize game
uno_deck = build_deck()
uno_deck = shuffle_deck(uno_deck)
discards = []

players_name = []
players = []
colors = ["Red", "Green", "Yellow", "Blue"]

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

print("\n=== Game Starting ===")
print(f"Players: {', '.join(players_name)}")

# Deal initial cards
for i in range(num_players):
    players.append(draw_cards(7, uno_deck, discards))
    print(f"{players_name[i]} received 7 cards.")

# Initialize game state
player_turn = 0
play_direction = 1  # 1 for clockwise, -1 for counter-clockwise
game_active = True

# Start with first card on discard pile
discards.append(uno_deck.pop(0))
top_card = discards[-1].split(" ", 1)
current_color = top_card[0]
current_value = top_card[1] if len(top_card) > 1 else "Any"

# Handle wild cards as starting card
if current_color == "Wild":
    print("Starting card is Wild. Choosing random color...")
    current_color = random.choice(colors)

print(f"\nGame begins with: {discards[-1]} ({current_color})")

# Main game loop
while game_active:
    current_hand = players[player_turn]
    
    # Show game status before each turn
    show_game_status()
    show_hand(player_turn, current_hand)
    print(f"Current card: {discards[-1]} ({current_color})")
    
    # Check if player can play
    if can_play(current_color, current_value, current_hand):
        print(f"Valid moves: {[i+1 for i, card in enumerate(current_hand) if 'Wild' in card or card.startswith(current_color) or current_value in card]}")
        
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
        print(f"\n{players_name[player_turn]} played: {played_card}")
        
        # Check if player won
        if not current_hand:
            game_active = False
            winner = players_name[player_turn]
            print("\n=== Game Over! ===")
            print(f"{winner} is the Winner!")
            break
        
        # Process special cards
        top_card = discards[-1].split(" ", 1)
        current_color = top_card[0]
        current_value = top_card[1] if len(top_card) > 1 else "Any"
        
        # Handle wild cards
        if current_color == "Wild":
            print("\n=== Wild Card ===")
            print("Choose a new color:")
            for i, color in enumerate(colors, 1):
                print(f"{i}) {color}")
            color_choice = get_valid_input("Enter color choice (1-4): ", 1, 4)
            current_color = colors[color_choice - 1]
            print(f"Color changed to {current_color}")
        
        # Handle action cards
        if current_value == "Reverse":
            play_direction *= -1
            print("\n=== Reverse ===")
            print("Direction reversed!")
        elif current_value == "Skip":
            skipped_player = (player_turn + play_direction) % num_players
            player_turn = skipped_player
            print("\n=== Skip ===")
            print(f"{players_name[skipped_player]} has been skipped!")
        elif current_value == "Draw Two":
            next_player = (player_turn + play_direction) % num_players
            drawn_cards = draw_cards(2, uno_deck, discards)
            players[next_player].extend(drawn_cards)
            print("\n=== Draw Two ===")
            print(f"{players_name[next_player]} draws 2 cards!")
        elif current_value == "Draw Four":
            next_player = (player_turn + play_direction) % num_players
            drawn_cards = draw_cards(4, uno_deck, discards)
            players[next_player].extend(drawn_cards)
            print("\n=== Draw Four ===")
            print(f"{players_name[next_player]} draws 4 cards!")
    else:
        print("\nYou can't play. Drawing a card...")
        drawn_card = draw_cards(1, uno_deck, discards)[0]
        players[player_turn].append(drawn_card)
        print(f"You drew: {drawn_card}")
        
        # Check if drawn card can be played
        card_color, card_value = drawn_card.split(" ", 1) if "Wild" not in drawn_card else (drawn_card, "")
        if "Wild" in drawn_card or card_color == current_color or card_value == current_value:
            print("You can play the drawn card!")
            play_choice = get_valid_input("Do you want to play it? (y/n): ", 0, 1, str)
            if play_choice == 'y':
                players[player_turn].remove(drawn_card)
                discards.append(drawn_card)
                print(f"\n{players_name[player_turn]} played: {drawn_card}")
                
                # Process special cards (similar to above)
                top_card = discards[-1].split(" ", 1)
                current_color = top_card[0]
                current_value = top_card[1] if len(top_card) > 1 else "Any"
                
                if current_color == "Wild":
                    print("\n=== Wild Card ===")
                    print("Choose a new color:")
                    for i, color in enumerate(colors, 1):
                        print(f"{i}) {color}")
                    color_choice = get_valid_input("Enter color choice (1-4): ", 1, 4)
                    current_color = colors[color_choice - 1]
                    print(f"Color changed to {current_color}")
                
                if current_value == "Reverse":
                    play_direction *= -1
                    print("\n=== Reverse ===")
                    print("Direction reversed!")
                elif current_value == "Skip":
                    skipped_player = (player_turn + play_direction) % num_players
                    player_turn = skipped_player
                    print("\n=== Skip ===")
                    print(f"{players_name[skipped_player]} has been skipped!")
                elif current_value == "Draw Two":
                    next_player = (player_turn + play_direction) % num_players
                    drawn_cards = draw_cards(2, uno_deck, discards)
                    players[next_player].extend(drawn_cards)
                    print("\n=== Draw Two ===")
                    print(f"{players_name[next_player]} draws 2 cards!")
                elif current_value == "Draw Four":
                    next_player = (player_turn + play_direction) % num_players
                    drawn_cards = draw_cards(4, uno_deck, discards)
                    players[next_player].extend(drawn_cards)
                    print("\n=== Draw Four ===")
                    print(f"{players_name[next_player]} draws 4 cards!")
    
    # Move to next player
    player_turn = (player_turn + play_direction) % num_players