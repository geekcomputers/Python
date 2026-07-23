import random

# Random number between 1 and 10
number = random.randint(1, 10)

# Make a loop
while True:

    try:
        # Ask for guess
        guess = int(input("Guess of a number (1, 10): "))

        if (guess < 1 or guess > 10):
            print("ERROR")
            continue;

        if (guess == number):
            print("You did it!")
            break
        elif (guess < number):
            print("Higher!")
        elif (guess > number):
            print("Lower!")
    # Robust error handling
    except (ValueError):
        print("ERROR")
