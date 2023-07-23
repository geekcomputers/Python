# importing the time module
import time

# importing the random module
import random

# welcoming the user
name = input("What is your name? ")

print("\nHello, " + name + "\nTime to play hangman!\n")

# wait for 1 second
time.sleep(1)

print("Start guessing...\nHint:It is a fruit")
time.sleep(0.5)

someWords = """apple banana mango strawberry  
orange grape pineapple apricot lemon coconut watermelon 
cherry papaya berry peach lychee muskmelon"""

someWords = someWords.split(" ")
# randomly choose a secret word from our "someWords" LIST.
word = random.choice(someWords)

# creates an variable with an empty value
guesses = ""

# determine the number of turns
turns = 5

# Create a while loop

# check if the turns are more than zero
while turns > 0:

    # make a counter that starts with zero
    failed = 0

    # for every character in secret_word
    for char in word:

        # see if the character is in the players guess
        if char in guesses:

            # print then out the character
            print(char, end=" ")

        else:

            # if not found, print a dash
            print("_", end=" ")

            # and increase the failed counter with one
            failed += 1

    # if failed is equal to zero

    # print You Won
    if failed == 0:
        print("\nYou won")

        # exit the script
        break

    print

    # ask the user go guess a character
    guess = input("\nGuess a character:")

    # Validation of the guess
    if not guess.isalpha():
        print("Enter only a LETTER")
        continue
    elif len(guess) > 1:
        print("Enter only a SINGLE letter")
        continue
    elif guess in guesses:
        print("You have already guessed that letter")
        continue

    # set the players guess to guesses
    guesses += guess

    # if the guess is not found in the secret word
    if guess not in word:

        # turns counter decreases with 1 (now 9)
        turns -= 1

        # print wrong
        print("\nWrong")

        # how many turns are left
        print("You have", +turns, "more guesses\n")

        # if the turns are equal to zero
        if turns == 0:

            # print "You Loose"
            print("\nYou Loose")
