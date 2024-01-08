import random
from colorama import Fore, Style
import inquirer

responses = [
    "It is certain",
    "It is decidedly so",
    "Without a doubt",
    "Yes definitely",
    "You may rely on it",
    "As I see it, yes",
    "Most likely",
    "Outlook good",
    "Yes",
    "Signs point to yes",
    "Do not count on it",
    "My reply is no",
    "My sources say no",
    "Outlook not so good",
    "Very doubtful",
    "Reply hazy try again",
    "Ask again later",
    "Better not tell you now",
    "Cannot predict now",
    "Concentrate and ask again",
]


# Will use a class on it.
# Will try to make it much more better.
def get_user_name():
    return inquirer.text(
        message="Hi! I am the magic 8 ball, what's your name?"
    ).execute()


def display_greeting(name):
    print(f"Hello, {name}!")


def magic_8_ball():
    question = inquirer.text(message="What's your question?").execute()
    answer = random.choice(responses)
    print(Fore.BLUE + Style.BRIGHT + answer + Style.RESET_ALL)
    try_again()


def try_again():
    response = inquirer.list_input(
        message="Do you want to ask more questions?",
        choices=["Yes", "No"],
    ).execute()

    if response.lower() == "yes":
        magic_8_ball()
    else:
        exit()


if __name__ == "__main__":
    user_name = get_user_name()
    display_greeting(user_name)
    magic_8_ball()
