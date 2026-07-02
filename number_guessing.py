import random

def guessing_game():
    print("Welcome to the Number Guessing Game!")

    upper_bound = int(input("Write the upper bound of the range: "))

    print(f"I'm thinking of a number between 1 and {upper_bound}.")
    
    number_to_guess = random.randint(1, upper_bound)
    attempts = 0
    guessed_correctly = False

    while not guessed_correctly:
        try:
            guess = int(input("Make a guess: "))
            attempts += 1
            
            if guess < number_to_guess:
                print("Too low.")
            elif guess > number_to_guess:
                print("Too high.")
            else:
                guessed_correctly = True
                print(f"Congratulations! You've guessed the number {number_to_guess} in {attempts} attempts.")
        except ValueError:
            print("Please enter a valid integer.")

if __name__ == "__main__":
    guessing_game()