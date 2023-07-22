from random import randint
from time import sleep


def guessing_game(GUESS_RANGE, GUESS_LIMIT):
    # Set the initial values.
    RANDOM = randint(1, GUESS_RANGE)
    GUESS = int(input("What is your guess? "))
    ATTEMPTS_ALLOWED = GUESS_LIMIT
    done = False

    # Validate the inputted guess.
    GUESS = InputValidation(GUESS, GUESS_RANGE)

    # Now we have a valid guess.
    while GUESS_LIMIT > 0 and not done:
        GUESS_LIMIT -= 1  # Take one guess = lose one chance
        if GUESS_LIMIT > 0:
            if GUESS < RANDOM:
                print(f"It should be higher than {GUESS}.")
            elif GUESS > RANDOM:
                print(f"It should be lower than {GUESS}.")
            else:
                ATTEMPTS_TOOK = ATTEMPTS_ALLOWED - GUESS_LIMIT
                print(f"You nailed it! And it only took you {ATTEMPTS_TOOK} attempts.")
                done = True
            if GUESS_LIMIT > 0 and not done:
                print(f"You still have {GUESS_LIMIT} chances left.\n")
                GUESS = int(input("Try a new guess: "))
                # Another input validation loop.
                GUESS = InputValidation(GUESS, GUESS_RANGE)
        elif GUESS_LIMIT == 0 and not done:  # Last chance to guess
            if GUESS == RANDOM:
                print(
                    f"You nailed it! However, it took you all the {ATTEMPTS_ALLOWED} attempts."
                )
            else:
                print(
                    f"GAME OVER! It took you more than {ATTEMPTS_ALLOWED} attempts. "
                    f"The correct number is {RANDOM}."
                )


def InputValidation(GUESS, GUESS_RANGE):
    while not 1 <= GUESS <= GUESS_RANGE:
        print("TRY AGAIN! Your guess is out of range!\n")
        GUESS = int(input("What is your guess? "))
    return GUESS


def easy():
    print("You are to guess a number between 1 and 10 in no more than 6 attempts.")
    guessing_game(10, 6)


def medium():
    print("You are to guess a number between 1 and 20 in no more than 4 attempts.")
    guessing_game(20, 4)


def hard():
    print("You are to guess a number between 1 and 50 in no more than 3 attempts.")
    guessing_game(50, 3)


def try_again():
    print()
    again = input("Do you want to play again? (yes/no) ")
    if again.lower() in ["y", "yes"]:
        welcome()
    elif again.lower() in ["n", "no"]:
        print("Thanks for playing the game")
    else:
        print("INVALID VALUE")
        try_again()


def welcome():
    print("Hello, Welcome to the Guessing Game!")
    name = input("I'm Geek! What's Your Name? ")
    sleep(1)

    print(f"Okay, {name}. Let's Begin The Guessing Game!")
    print(
        "Choose a level:",
        "1. Easy",
        "2. Medium",
        "3. Hard",
        sep="\n",
    )
    sleep(1)
    level = int(input("Pick a number: "))
    print()
    sleep(1)
    if level == 1:
        easy()
        try_again()
    elif level == 2:
        medium()
        try_again()
    elif level == 3:
        hard()
        try_again()
    else:
        print("INVALID VALUE! Please try again.\n")
        welcome()


welcome()
