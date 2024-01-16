import os
import sys
from pprint import pprint

import sys

sys.path.append(os.path.realpath("."))
import inquirer  # noqa

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
    def phone_validation():
        # Think over how to make a validation for phone number?
        pass

    def email_validation():
        pass

    def password_validation():
        pass

    def username_validation():
        pass

    def country_validation():
        # All the countries in the world???
        # JSON can be used.
        # Download the file

    def state_validation():
        # All the states in the world??
        # The state of the selected country only.
        pass

    def city_validation():
        # All the cities in the world??
        # JSON can be used.
        pass


# Have an option to go back.
# How can I do it?
if answers["authentication"] == "Login":
    print("Login")
    questions = [
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


elif answers["authentication"] == "Sign up":
    print("Sign up")

    questions = [
        inquirer.Text(
            "name",
            message="What's your first name?",
            validate=Validation.fname_validation,
        ),
        inquirer.Text(
            "surname",
            message="What's your last name(surname)?, validate=Validation.lname), {name}?",
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
            "password",
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
            "state",
            message="What's your state",
            validate=Validation.state_validation,
        ),
        inquirer.Text(
            "city",
            message="What's your city",
            validate=Validation.city_validation,
        ),
        inquirer.Text(
            "address",
            message="What's your address",
            validate=Validation.address_validation,
        ),
    ]
# Also add optional in the above thing.
# Have string manipulation for the above thing.
# How to add authentication of google to command line?
elif answers["authentication"] == "Exit":
    print("Exit")
    sys.exit()

pprint(answers)
