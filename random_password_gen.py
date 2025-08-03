"""
random_password_gen.py
A script to generate strong random passwords.

Usage:
$ python random_password_gen.py

Author: Keshavraj Pore
"""

import random
import string


def generate_password(length=12):
    characters = string.ascii_letters + string.digits + string.punctuation
    password = "".join(random.choice(characters) for _ in range(length))
    return password


def main():
    print("Random Password Generator")
    try:
        length = int(input("Enter desired password length: "))
        if length < 6:
            print(" Password length should be at least 6.")
            return
        password = generate_password(length)
        print(f"\nGenerated Password: {password}")

        # Save to file
        with open("passwords.txt", "a") as file:
            file.write(password + "\n")
        print(" Password saved to passwords.txt")

    except ValueError:
        print(" Please enter a valid number.")


if __name__ == "__main__":
    main()
