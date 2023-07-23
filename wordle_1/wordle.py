# Get all 5 letter words from the full English dictionary
"""
# dictionary by http://www.gwicks.net/dictionaries.htm
# Load full English dictionary
dictionary = open("Dictionary.txt", 'r')
# Load new empty dictionary
new_dictionary = open("5 letter word dictionary.txt", "w")

# Read the full English dictionary
dictionary_content = dictionary.read()
# Split the full dictionary on every new line
dictionary_content = dictionary_content.split("\n") # This returns a list of all the words in the dictionary

# Loop over all the words in the full dictionary
for i in dictionary_content:
    # Check if the current word is 5 characters long
    if len(i) == 5:
        # Write word to the new dictionary
        new_dictionary.write(f"{i}\n")

# Close out of the new dictionary
new_dictionary.close()
"""

# import the library random
import random

# Load 5 letter word dictionary
with open("5 letter word dictionary.txt", 'r') as dictionary:
    # Read content of dictionary
    dictionary = dictionary.read().split('\n') # This returns a list of all the words in the dictionary

# Choose a random word from the dictionary
word = random.choice(dictionary)

# Get all the unique letters of the word
dif_letters = list(set(word))

# Count how many times each letter occurs in the word
count_letters = {}
for i in dif_letters:
    count_letters[i] = word.count(i)

# Set tries to 0
tries = 0

# Main loop
while True:
    # Check if the user has used all of their tries
    if tries == 6:
        print(f"You did not guess the word!\nThe word was {word}")
        break
    # Get user input and make it all lower case
    user_inp = input(">>").lower()

    # Check if user wants to exit the program
    if user_inp == "q":
        break

    # Check if the word given by the user is 5 characters long
    if not len(user_inp) == 5:
        print("Your input must be 5 letters long")
        continue

    # Check if the word given by the user is in the dictionary
    if not user_inp in dictionary:
        print("Your word is not in the dictionary")
        continue

    # Check if the word given by the user is correct
    if user_inp == word:
        print(f"You guessed the word in {tries} tries")
        break

    # Check guess
    letter = 0
    letter_dict = {}
    letters_checked = []
    return_answer = "  "
    for i in word:
        # Check if letter is already checked
        counter = 0
        cont = False
        for g in letters_checked:
            if g == user_inp[letter]:
                counter += 1
                # Check if letter has been checkd more or equal to the ammount of these letters inside of the word
                if counter >= count_letters[i]:
                    cont = True

        # Check if cont is true
        if cont:
            return_answer += "-"
            letters_checked.append(user_inp[letter])
            letter += 1
            continue


        answer_given = False
        do_not_add = False
        # Check if letter is in word
        if user_inp[letter] in word:
            answer_given = True
            # Check if letter is in the correct position
            if user_inp[letter] == i:
                return_answer += "G"
            else:
                if not user_inp[word.index(user_inp[letter])] == word[word.index(user_inp[letter])]:
                    return_answer += "Y"
                else:
                    answer_given = False
                    do_not_add = True

        # Check if there has already been an answer returned
        if not answer_given:
            return_answer += "-"

        # Append checked letter to the list letters_checked
        if not do_not_add:
           letters_checked.append(user_inp[letter])

        letter += 1

    print(return_answer)

    tries += 1
