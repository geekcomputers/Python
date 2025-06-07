# PasswordGenerator GGearing 314 01/10/19
# modified Prince Gangurde 4/4/2020

import random
import pycountry

def generate_password():
    # Define characters and word sets
    special_characters = list("!@#$%/?<>|&*-=+_")
    
    animals = (
        "ant", "alligator", "baboon", "badger", "barb", "bat", "beagle", "bear", "beaver", "bird",
        "bison", "bombay", "bongo", "booby", "butterfly", "bee", "camel", "cat", "caterpillar",
        "catfish", "cheetah", "chicken", "chipmunk", "cow", "crab", "deer", "dingo", "dodo", "dog",
        "dolphin", "donkey", "duck", "eagle", "earwig", "elephant", "emu", "falcon", "ferret", "fish",
        "flamingo", "fly", "fox", "frog", "gecko", "gibbon", "giraffe", "goat", "goose", "gorilla"
    )

    colours = (
        "red", "orange", "yellow", "green", "blue", "indigo", "violet", "purple",
        "magenta", "cyan", "pink", "brown", "white", "grey", "black"
    )

    # Get random values
    animal = random.choice(animals)
    colour = random.choice(colours)
    number = random.randint(1, 999)
    special = random.choice(special_characters)
    case_choice = random.choice(["upper_colour", "upper_animal"])

    # Pick a random country and language
    country = random.choice(list(pycountry.countries)).name
    languages = [lang.name for lang in pycountry.languages if hasattr(lang, "name")]
    language = random.choice(languages)

    # Apply casing
    if case_choice == "upper_colour":
        colour = colour.upper()
    else:
        animal = animal.upper()

    # Combine to form password
    password = f"{colour}{number}{animal}{special}"
    print("Generated Password:", password)
    print("Based on Country:", country)
    print("Language Hint:", language)

# Run it
generate_password()
