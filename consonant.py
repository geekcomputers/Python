my_string = input("Enter a string to count number of consonants: ")
string_check = [
    "a",
    "e",
    "i",
    "o",
    "u",
    "A",
    "E",
    "I",
    "O",
    "U",
]  # list for checking vowels


def count_con(string):
    c = 0
    for i in range(len(string)):
        if (
            string[i] not in string_check
        ):  # counter increases if the character is not vowel
            c += 1
    return c


counter = count_con(my_string)
print(f"Number of consonants in {my_string} is {counter}.")
