"""
This program converts English text to Pig-Latin. In Pig-Latin, we take the first letter of each word, 
move it to the end, and add 'ay'. If the first letter is a vowel, we simply add 'hay' to the end. 
The program preserves capitalization and title case.

For example:
- "Hello" becomes "Ellohay"
- "Image" becomes "Imagehay"
- "My name is John Smith" becomes "Ymay amenay ishay Ohnjay Mithsmay"
"""


def pig_latin_word(word):
    vowels = "AEIOUaeiou"

    if word[0] in vowels:
        return word + "hay"
    else:
        return word[1:] + word[0] + "ay"

def pig_latin_sentence(text):
    words = text.split()
    pig_latin_words = []

    for word in words:
        # Preserve capitalization
        if word.isupper():
            pig_latin_words.append(pig_latin_word(word).upper())
        elif word.istitle():
            pig_latin_words.append(pig_latin_word(word).title())
        else:
            pig_latin_words.append(pig_latin_word(word))

    return ' '.join(pig_latin_words)

user_input = input("Enter some English text: ")
pig_latin_text = pig_latin_sentence(user_input)
print("\nPig-Latin: " + pig_latin_text)
