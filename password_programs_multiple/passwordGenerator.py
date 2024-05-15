# PasswordGenerator GGearing 314 01/10/19
# modified Prince Gangurde 4/4/2020

from random import randint
import pycountry

case = randint(1, 2)
number = randint(1, 999)

# TODO: Pick random country from it

countries = list(pycountry.countries)
country_names = [country.name for country in countries]

print(country_names)

# TODO: Try to add languages, too.

specialCharacters = (
    "!",
    "@",
    "#",
    "$",
    "%",
    "/",
    "?",
    ":",
    "<",
    ">",
    "|",
    "&",
    "*",
    "-",
    "=",
    "+",
    "_",
)

animals = (
    "ant",
    "alligator",
    "baboon",
    "badger",
    "barb",
    "bat",
    "beagle",
    "bear",
    "beaver",
    "bird",
    "bison",
    "bombay",
    "bongo",
    "booby",
    "butterfly",
    "bee",
    "camel",
    "cat",
    "caterpillar",
    "catfish",
    "cheetah",
    "chicken",
    "chipmunk",
    "cow",
    "crab",
    "deer",
    "dingo",
    "dodo",
    "dog",
    "dolphin",
    "donkey",
    "duck",
    "eagle",
    "earwig",
    "elephant",
    "emu",
    "falcon",
    "ferret",
    "fish",
    "flamingo",
    "fly",
    "fox",
    "frog",
    "gecko",
    "gibbon",
    "giraffe",
    "goat",
    "goose",
    "gorilla",
)

colour = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "indigo",
    "violet",
    "purple",
    "magenta",
    "cyan",
    "pink",
    "brown",
    "white",
    "grey",
    "black",
)

chosenanimal = animals[
    randint(0, len(animals) - 1)
]  # randint will return max lenght but , tuple has index from 0 to len-1
chosencolour = colour[randint(0, len(colour) - 1)]
chosenSpecialCharacter = specialCharacters[randint(0, len(specialCharacters) - 1)]

if case == 1:
    chosenanimal = chosenanimal.upper()
    print(chosencolour + str(number) + chosenanimal + chosenSpecialCharacter)
else:
    chosencolour = chosencolour.upper()
    print(chosenanimal + str(number) + chosencolour + chosenSpecialCharacter)

# Try to consolidate unify the characters.


# The program can be further improved.
