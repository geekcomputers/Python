import re

def phone_validation(phone_number):
    # Match a typical US phone number format (xxx) xxx-xxxx
    pattern = re.compile(r'^\(\d{3}\) \d{3}-\d{4}$')
    return bool(pattern.match(phone_number))

# Example usage:
phone_number_input = input("Enter phone number: ")
if phone_validation(phone_number_input):
    print("Phone number is valid.")
else:
    print("Invalid phone number.")

def email_validation(email):
    # Basic email format validation
    pattern = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
    return bool(pattern.match(email))

# Example usage:
email_input = input("Enter email address: ")
if email_validation(email_input):
    print("Email address is valid.")
else:
    print("Invalid email address.")


def password_validation(password):
    # Password must be at least 8 characters long and contain at least one digit
    return len(password) >= 8 and any(char.isdigit() for char in password)

# Example usage:
password_input = input("Enter password: ")
if password_validation(password_input):
    print("Password is valid.")
else:
    print("Invalid password.")


def username_validation(username):
    # Allow only alphanumeric characters and underscores
    return bool(re.match('^[a-zA-Z0-9_]+$', username))

# Example usage:
username_input = input("Enter username: ")
if username_validation(username_input):
    print("Username is valid.")
else:
    print("Invalid username.")


def country_validation(country):
    # Example: Allow only alphabetical characters and spaces
    return bool(re.match('^[a-zA-Z ]+$', country))

# Example usage:
country_input = input("Enter country name: ")
if country_validation(country_input):
    print("Country name is valid.")
else:
    print("Invalid country name.")

