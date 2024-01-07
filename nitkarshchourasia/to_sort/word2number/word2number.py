def word_to_number(word):
    numbers_dict = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
        "thousand": 1000,
        "lakh": 100000,
        "crore": 10000000,
        "billion": 1000000000,
        "trillion": 1000000000000,
    }

    # Split the string into words
    words = word.split()

    result = 0
    current_number = 0

    # Ways I can make this more efficient:
    for w in words:
        if w in numbers_dict:
            current_number += numbers_dict[w]
        elif w == "hundred":
            current_number *= 100
        elif w == "thousand":
            result += current_number * 1000
            current_number = 0
        elif w == "lakh":
            result += current_number * 100000
            current_number = 0
        elif w == "crore":
            result += current_number * 10000000
            current_number = 0
        elif w == "billion":
            result += current_number * 1000000000
            current_number = 0
        elif w == "trillion":
            result += current_number * 1000000000000
            current_number = 0

    result += current_number

    return result


# Example usage:
number_str = "two trillion seven billion fifty crore thirty-four lakh seven thousand nine hundred"
result = word_to_number(number_str)
print(result)


# Will make a tkinter application out of it.
## It will have a slider to use the more efficient way or just the normal way.
## More efficient way would have a library word2num to choose from.

# The application would be good.
# I want to make it more efficient and optimized.
