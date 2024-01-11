# This is like making a package.lock file for npm package.
# Yes, I should be making it.
__author__ = "Nitkarsh Chourasia"
__version__ = "0.0.0"  # SemVer # Understand more about it
__license__ = "MIT"  # Understand more about it
# Want to make it open source but how to do it?
# Program to make a simple calculator
# Will have to extensively work on Jarvis and local_document and MongoDb and Redis and JavaScript and CSS and DOM manipulation to understand it.
# Will have to study maths to understand it more better.
# How can I market gtts? Like showing used google's api? This is how can I market it?
# Project description? What will be the project description?

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


# Find the best of best extensions for the auto generation of the documentation parts.
# For your favourite languages like JavaScript, Python ,etc,...
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
        """summary: Get the sum of numbers

        Returns:
            _type_: _description_
        """
        return self.num1 + self.num2

    def sub(self):
        """_summary_: Get the difference of numbers

        Returns:
            _type_: _description_
        """
        return self.num1 - self.num2

    def multi(self):
        """_summary_: Get the product of numbers

        Returns:
            _type_: _description_
        """
        return self.num1 * self.num2

    def div(self):
        """_summary_: Get the quotient of numbers

        Returns:
            _type_: _description_
        """
        # What do we mean by quotient?
        return self.num1 / self.num2

    def power(self):
        """_summary_: Get the power of numbers

        Returns:
            _type_: _description_
        """
        return self.num1**self.num2

    def root(self):
        """_summary_: Get the root of numbers

        Returns:
            _type_: _description_
        """
        return self.num1 ** (1 / self.num2)

    def remainer(self):
        """_summary_: Get the remainder of numbers

        Returns:
            _type_: _description_
        """

        # Do I have to use the '.' period or full_stop in the numbers?
        return self.num1 % self.num2

    def cube_root(self):
        """_summary_: Get the cube root of numbers

        Returns:
            _type_: _description_
        """
        return self.num1 ** (1 / 3)

    def cube_exponent(self):
        """_summary_: Get the cube exponent of numbers

        Returns:
            _type_: _description_
        """
        return self.num1**3

    def square_root(self):
        """_summary_: Get the square root of numbers

        Returns:
            _type_: _description_
        """
        return self.num1 ** (1 / 2)

    def square_exponent(self):
        """_summary_: Get the square exponent of numbers

        Returns:
            _type_: _description_
        """
        return self.num1**2

    def factorial(self):
        """_summary_: Get the factorial of numbers"""
        pass

    def list_factors(self):
        """_summary_: Get the list of factors of numbers"""
        pass

    def factorial(self):
        for i in range(1, self.num + 1):
            self.factorial = self.factorial * i  # is this right?

    def LCM(self):
        """_summary_: Get the LCM of numbers"""
        pass

    def HCF(self):
        """_summary_: Get the HCF of numbers"""
        pass

    # class time: # Working with days calculator
    def age_calculator(self):
        """_summary_: Get the age of the user"""
        # This is be very accurate and precise it should include proper leap year and last birthday till now every detail.
        # Should show the preciseness in seconds when called.
        pass

    def days_calculator(self):
        """_summary_: Get the days between two dates"""
        pass

    def leap_year(self):
        """_summary_: Get the leap year of the user"""
        pass

    def perimeter(self):
        """_summary_: Get the perimeter of the user"""
        pass

    class Trigonometry:
        """_summary_: Class enriched with all the methods to solve basic trignometric problems"""

        def pythagorean_theorem(self):
            """_summary_: Get the pythagorean theorem of the user"""
            pass

        def find_hypotenuse(self):
            """_summary_: Get the hypotenuse of the user"""
            pass

        def find_base(self):
            """_summary_: Get the base of the user"""
            pass

        def find_perpendicular(self):
            """_summary_: Get the perpendicular of the user"""
            pass

    # class Logarithms:
    # Learn more about Maths in general

    def quadratic_equation(self):
        """_summary_: Get the quadratic equation of the user"""
        pass

    def open_system_calculator(self):
        """_summary_: Open the calculator present on the machine of the user"""
        # first identify the os
        # track the calculator
        # add a debugging feature like error handling
        # for linux and mac
        # if no such found then print a message to the user that sorry dear it wasn't possible to so
        # then open it

    def take_inputs(self):
        """_summary_: Take the inputs from the user in proper sucession"""
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
        """_summary_: Greet the user with using Audio"""
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
        """_summary_: Get the name of the user and have an option to greet him/her"""
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
        """_summary_: Get the name of the user and have an option to show him his user name in art"""
        # Default is to show = True, else False if user tries to disable it.

        # Tell him to show the time and date
        # print(art.text2art(self.name))
        # print(date and time of now)
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

    class unitConversion:
        """_summary_: Class enriched with all the methods to convert units"""

        # Do we full-stops in generating documentations?

        def __init__(self):
            """_summary_: Initialise the class with the required attributes"""
            self.take_inputs()

        def length(self):
            """_summary_: Convert length units"""
            # It should have a meter to unit and unit to meter converter
            # Othe lengths units it should also have.
            # Like cm to pico meter and what not
            pass

        def area(self):
            # This will to have multiple shapes and polygons to it to improve it's area.
            # This will to have multiple shapes and polygons to it to improve it's area.
            # I will try to use the best of the formula to do it like the n number of polygons to be solved.

            pass

        def volume(self):
            # Different shapes and polygons to it to improve it's volume.
            pass

        def mass(self):
            pass

        def time(self):
            pass

        def speed(self):
            pass

        def temperature(self):
            pass

        def data(self):
            pass

        def pressure(self):
            pass

        def energy(self):
            pass

        def power(self):
            pass

        def angle(self):
            pass

        def force(self):
            pass

        def frequency(self):
            pass

        def take_inputs(self):
            pass

    class CurrencyConverter:
        def __init__(self):
            self.take_inputs()

        def take_inputs(self):
            pass

        def convert(self):
            pass

    class Commands:
        def __init__(self):
            self.take_inputs()

        def previous_number(self):
            pass

        def previous_operation(self):
            pass

        def previous_result(self):
            pass

    def clear_screen(self):
        # Do I need a clear screen?
        # os.system("cls" if os.name == "nt" else "clear")
        # os.system("cls")
        # os.system("clear")
        pass


if __name__ == "__main__":
    operation_1 = Calculator(10, 5)

    # Operations
    # User interaction
    # Study them properly and try to understand them.
    # Study them properly and try to understand them in very detailed length. Please.
    # Add a function to continually ask for input until the user enters a valid input.


# Let's explore colorma
# Also user log ins, and it saves user data and preferences.
# A feature of the least priority right now.

# List of features priority should be planned.


# Documentations are good to read and understand.
# A one stop solution is to stop and read the document.
# It is much better and easier to understand.
