import random

# Define the game board with snakes and ladders
snakes_and_ladders = {
    2: 38, 7: 14, 8: 31, 15: 26, 16: 6, 21: 42,
    28: 84, 36: 44, 46: 25, 49: 11, 51: 67, 62: 19,
    64: 60, 71: 91, 74: 53, 78: 98, 87: 94, 89: 68,
    92: 88, 95: 75, 99: 80
}

# Function to roll a six-sided die
def roll_die():
    return random.randint(1, 6)

# Function to simulate a single turn
def take_turn(current_position, player_name):
    # Roll the die
    roll_result = roll_die()
    print(f"{player_name} rolled a {roll_result}!")

    # Calculate the new position after the roll
    new_position = current_position + roll_result

    # Check if the new position is a ladder or a snake
    if new_position in snakes_and_ladders:
        new_position = snakes_and_ladders[new_position]
        if new_position > current_position:
            print("Ladder! Climb up!")
        else:
            print("Snake! Slide down!")

    # Check if the new position exceeds the board size
    if new_position >= 100:
        new_position = 100
        print(f"Congratulations, {player_name} reached the final square!")

    return new_position

# Main game loop
def play_snakes_and_ladders():
    player1_position = 1
    player2_position = 1

    player1_name = input("Enter the name of Player 1: ")
    player2_name = input("Enter the name of Player 2: ")

    current_player = player1_name

    while player1_position < 100 and player2_position < 100:
        print(f"\n{current_player}'s turn:")
        input("Press Enter to roll the die.")

        if current_player == player1_name:
            player1_position = take_turn(player1_position, player1_name)
            current_player = player2_name
        else:
            player2_position = take_turn(player2_position, player2_name)
            current_player = player1_name

    print("\nGame Over!")
    print(f"{player1_name} ended at square {player1_position}.")
    print(f"{player2_name} ended at square {player2_position}.")
    if player1_position == 100:
        print(f"{player1_name} won!")
    elif player2_position == 100:
        print(f"{player2_name} won!")

# Start the game
play_snakes_and_ladders()
