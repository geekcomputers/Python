"""
Rock, Paper, Scissors Game 
Author: DEVANSH-GAJJAR
"""

import random


def get_user_choice():
    """Prompt the user to enter their choice."""
    choice = input("Enter your choice (rock, paper, scissors): ").lower()
    if choice in ["rock", "paper", "scissors"]:
        return choice
    else:
        print("Invalid choice! Please enter rock, paper, or scissors.")
        return get_user_choice()


def get_computer_choice():
    """Randomly select computer's choice."""
    options = ["rock", "paper", "scissors"]
    return random.choice(options)


def decide_winner(player, computer):
    """Decide the winner based on the choices."""
    if player == computer:
        return "It's a draw!"
    elif (
        (player == "rock" and computer == "scissors")
        or (player == "paper" and computer == "rock")
        or (player == "scissors" and computer == "paper")
    ):
        return "You win!"
    else:
        return "Computer wins!"


def main():
    """Main function to play the game."""
    user_choice = get_user_choice()
    computer_choice = get_computer_choice()
    print(f"Computer chose: {computer_choice}")
    print(decide_winner(user_choice, computer_choice))


if __name__ == "__main__":
    main()
