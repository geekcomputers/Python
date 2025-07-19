import random

# Game state variables
players = {}  # Stores player names and their positions
is_ready = {}  # Tracks if player has rolled a 6 to start
current_position = 1  # Initial position for new players
game_active = True  # Controls the main game loop


def get_valid_integer(prompt: str, min_value: int = None, max_value: int = None) -> int:
    """
    Get a valid integer input from the user within a specified range.

    Args:
        prompt: The message to display to the user.
        min_value: The minimum acceptable value (inclusive).
        max_value: The maximum acceptable value (inclusive).

    Returns:
        A valid integer within the specified range.
    """
    while True:
        try:
            value = int(input(prompt))
            if (min_value is not None and value < min_value) or (
                max_value is not None and value > max_value
            ):
                print(f"Please enter a number between {min_value} and {max_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def initialize_players() -> None:
    """Initialize players for the game"""
    global players, is_ready, current_position

    while True:
        player_count = get_valid_integer("Enter the number of players: ", min_value=1)

        for i in range(player_count):
            name = input(f"Enter player {i + 1} name: ").strip()
            if not name:
                name = f"Player {i + 1}"
            players[name] = current_position
            is_ready[name] = False

        start_game()
        break


def roll_dice() -> int:
    """Roll a 6-sided dice"""
    return random.randint(1, 6)


def start_game() -> None:
    """Start the main game loop"""
    global game_active, players, is_ready

    while game_active:
        print("/" * 20)
        print("1 -> Roll the dice")
        print("2 -> Start new game")
        print("3 -> Exit the game")
        print("/" * 20)

        for player in players:
            if not game_active:
                break

            choice = (
                input(
                    f"{player}'s turn (press Enter to roll or enter option): "
                ).strip()
                or "1"
            )

            try:
                choice = int(choice)
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            if players[player] < 100:
                if choice == 1:
                    dice_roll = roll_dice()
                    print(f"You rolled a {dice_roll}")

                    # Check if player can start moving
                    if not is_ready[player] and dice_roll == 6:
                        is_ready[player] = True
                        print(f"{player} can now start moving!")

                    # Process move if player is active
                    if is_ready[player]:
                        total_move = dice_roll
                        consecutive_sixes = 0

                        # Handle consecutive sixes
                        while dice_roll == 6 and consecutive_sixes < 2:
                            consecutive_sixes += 1
                            print("You rolled a 6! Roll again...")
                            dice_roll = roll_dice()
                            print(f"You rolled a {dice_roll}")
                            total_move += dice_roll

                        # Check for three consecutive sixes penalty
                        if consecutive_sixes == 2:
                            print("Three consecutive sixes! You lose your turn.")
                            total_move = 0

                        # Calculate new position
                        new_position = players[player] + total_move

                        # Validate move
                        if new_position > 100:
                            new_position = 100 - (new_position - 100)
                            print(f"Overshot! You bounce back to {new_position}")
                        elif new_position == 100:
                            print(f"Congratulations, {player}! You won the game!")
                            game_active = False
                            return

                        # Apply snakes and ladders
                        players[player] = new_position
                        players[player] = check_snakes(players[player], player)
                        players[player] = check_ladders(players[player], player)

                        print(f"{player} is now at position {players[player]}")

                elif choice == 2:
                    # Reset game state
                    players = {}
                    is_ready = {}
                    current_position = 1
                    initialize_players()
                    return

                elif choice == 3:
                    print("Thanks for playing! Goodbye.")
                    game_active = False
                    return

                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            else:
                print(f"{player} has already won the game!")


def check_snakes(position: int, player: str) -> int:
    """Check if the player landed on a snake"""
    snakes = {32: 10, 36: 6, 48: 26, 63: 18, 88: 24, 95: 56, 97: 78}

    if position in snakes:
        new_position = snakes[position]
        print(f"Snake bite! {player} slides down from {position} to {new_position}")
        return new_position
    return position


def check_ladders(position: int, player: str) -> int:
    """Check if the player landed on a ladder"""
    ladders = {4: 14, 8: 30, 20: 38, 28: 76, 40: 42, 50: 67, 71: 92, 80: 99}

    if position in ladders:
        new_position = ladders[position]
        print(f"Ladder! {player} climbs from {position} to {new_position}")
        return new_position
    return position


if __name__ == "__main__":
    print("/" * 40)
    print("Welcome to the Snake and Ladder Game!")
    print("/" * 40)
    initialize_players()
