# Program to make a simple calculator

from numbers import Number
from sys import exit
import colorama as color
import inquirer
from gtts import gTTS
from pygame import mixer, time
from io import BytesIO
from pprint import pprint
import art
import date


# Should be able to print date and time too.
# Should use voice assistant for specially abled people.
# A fully personalised calculator.
# voice_assistant on/off , setting bool value to true or false

# Is the operations valid?


# Validation checker
class Calculator:
    def __init__(self):
        self.take_inputs()

    def add(self):
        return self.num1 + self.num2

    def sub(self):
        return self.num1 - self.num2

    def multi(self):
        return self.num1 * self.num2

    def div(self):
        return self.num1 / self.num2

    def power(self):
        return self.num1**self.num2

    def root(self):
        return self.num1 ** (1 / self.num2)

    def remainer(self):
        return self.num1 % self.num2

    def cube_root(self):
        return self.num1 ** (1 / 3)

    def cube_exponent(self):
        return self.num1**3

    def square_root(self):
        return self.num1 ** (1 / 2)

    def square_exponent(self):
        return self.num1**2

    def take_inputs(self):
        while True:
            while True:
                try:
                    # self.num1 = float(input("Enter The First Number: "))
                    # self.num2 = float(input("Enter The Second Number: "))
                    pprint("Enter your number")
                    # validation check must be done
                    break
                except ValueError:
                    pprint("Please Enter A Valid Number")
                    continue
                # To let the user to know it is time to exit.
            pprint("Press 'q' to exit")
        # if self.num1 == "q" or self.num2 == "q":
        #     exit()  # Some how I need to exit it

    def greeting(self):
        text_to_audio = "Welcome To The Calculator"
        self.gtts_object = gTTS(text=text_to_audio, lang="en", tld="co.in", slow=False)
        tts = self.gtts_object
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)  # Reset the BytesIO object to the beginning
        mixer.init()
        mixer.music.load(fp)
        mixer.music.play()
        while mixer.music.get_busy():
            time.Clock().tick(10)

    # Here OOP is not followed.
    def user_name(self):
        self.name = input("Please enter your good name: ")
        # Making validation checks here
        text_to_audio = "{self.name}"
        self.gtts_object = gTTS(text=text_to_audio, lang="en", tld="co.in", slow=False)
        tts = self.gtts_object
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)  # Reset the BytesIO object to the beginning
        mixer.init()
        mixer.music.load(fp)
        mixer.music.play()
        while mixer.music.get_busy():
            time.Clock().tick(10)

    def user_name_art(self):
        # Remove whitespaces from beginning and end
        # Remove middle name and last name
        # Remove special characters
        # Remove numbers
        f_name = self.name.split(" ")[0]
        f_name = f_name.strip()
        # Remove every number present in it
        # Will have to practice not logic
        f_name = "".join([i for i in f_name if not i.isdigit()])

        # perform string operations on it for the art to be displayed.
        # Remove white spaces
        # Remove middle name and last name
        # Remove special characters
        # Remove numbers
        # Remove everything


if __name__ == "__main__":
    operation_1 = Calculator(10, 5)

    # Operations
    # User interaction
    # Study them properly and try to understand them.
    # Study them properly and try to understand them in very detailed length. Please.
    # Add a function to continually ask for input until the user enters a valid input.


# Let's explore colorma
