import os
import sys
from pprint import pprint


sys.path.append(os.path.realpath("."))
import inquirer

# Take authentication input from the user
questions = [
    inquirer.List(
        "authentication",  # This is the key
        message="Choose an option",
        choices=["Login", "Sign up", "Exit"],
    ),
]
answers = inquirer.prompt(questions)


# Just making pipelines
class Validation:
    @staticmethod
    def phone_validation(answer, current):
        # Think over how to make a validation for phone number?
        return True

    @staticmethod
    def email_validation(answer, current):
        return True

    @staticmethod
    def password_validation(answer, current):
        return True

    @staticmethod
    def username_validation():
        pass

    @staticmethod
    def fname_validation(answer, current):
        # Add your first name validation logic here
        return True

    @staticmethod
    def lname_validation(answer, current):
        # Add your last name validation logic here
        return True

    @staticmethod
    def country_validation(answer, current):
        # All the countries in the world???
        # JSON can be used.
        # Download the file
        return True

    @staticmethod
    def state_validation(answer, current):
        # All the states in the world??
        # The state of the selected country only.
        return True

    @staticmethod
    def city_validation(answer, current):
        # All the cities in the world??
        # JSON can be used.
        return True

    @staticmethod
    def password_confirmation(answer, current):
        return True

    @staticmethod
    def address_validation(answer, current):
        return True

    @staticmethod
    def login_username(answer, current):
        # Add your username validation logic here
        return True

    @staticmethod
    def login_password(answer, current):
        # Add your password validation logic here
        return True


# Have an option to go back.
# How can I do it?
if answers is not None and answers.get("authentication") == "Login":
    questions = [
        inquirer.Text(
            "surname",
            message="What's your last name (surname)?",
            validate=Validation.lname_validation,
        ),
        inquirer.Text(
            "username",
            message="What's your username?",
            validate=Validation.login_username,
        ),
        inquirer.Text(
            "password",
            message="What's your password?",
            validate=Validation.login_password,
        ),
    ]
    answers = inquirer.prompt(questions)

elif answers is not None and answers.get("authentication") == "Sign up":
    print("Sign up")
    questions = [
        inquirer.Text(
            "name",
            message="What's your first name?",
            validate=Validation.fname_validation,
        ),
        inquirer.Text(
            "surname",
            message="What's your last name (surname)?",
            validate=Validation.lname_validation,
        ),
        inquirer.Text(
            "phone",
            message="What's your phone number",
            validate=Validation.phone_validation,
        ),
        inquirer.Text(
            "email",
            message="What's your email",
            validate=Validation.email_validation,
        ),
        inquirer.Text(
            "password",
            message="What's your password",
            validate=Validation.password_validation,
        ),
        inquirer.Text(
            "password_confirm",
            message="Confirm your password",
            validate=Validation.password_confirmation,
        ),
        inquirer.Text(
            "username",
            message="What's your username",
            validate=Validation.username_validation,
        ),
        inquirer.Text(
            "country",
            message="What's your country",
            validate=Validation.country_validation,
        ),
        inquirer.Text(
            "address",
            message="What's your address",
            validate=Validation.address_validation,
        ),
    ]
    answers = inquirer.prompt(questions)

elif answers is not None and answers.get("authentication") == "Exit":
    print("Exit")
    sys.exit()

pprint(answers)
