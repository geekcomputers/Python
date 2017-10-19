'''Author Anurag Kumar(mailtoanuragkumarak95@gmail.com)
Module for implementing the simpest Magic 8 Ball Game.

Python:
  - 3.5

Requirements:
  - colorama

Usage:
  - $python3 magic8ball.py

Ask a question, and know the future.
'''
from time import sleep
from random import randint
from colorama import Fore, Style

# response list..
response = [
    "It is certain",
    "It is decidedly so",
    "Without a doubt",
    "Yes, definitely",
    "You may rely on it",
    "As I see it, yes",
    "Most likely",
    "Outlook good",
    "Yes",
    "Signs point to yes",
    "Quite possibly so",
    "Ask again later",
    "Better not tell you now",
    "Cannot predict now",
    "Concentrate and ask again",
    "Don't count on it",
    "My reply is no",
    "My sources say no",
    "Outlook not so good",
    "Very doubtful"]

# core game...
def game():
    ques = str(input("What is your question? \n").lower())
    print ("thinking...")
    sleep(1)
    idx = randint(0,20)
    if idx <10: color = Fore.GREEN
    elif idx>=10 and idx<15: color = Fore.YELLOW
    else: color = Fore.RED
    print (color+response[idx]+Style.RESET_ALL+'\n\n')
    playloop()

# looping func...
def playloop():
    ques_again = str(input("Would you like to ask another question? (y/n)\n").lower())
    if ques_again == 'y':
        game()

    elif ques_again == 'n':
        print("Auf Wiedersehen!")

    else:
        print ("What was that?/n")
        playloop()

if __name__=='__main__':
    game()
