print("\n### Vowel counter ###\n")
string = input("Enter a string: ").lower()
vowels = ["a", "e", "i", "o", "u"]

vowelscounter = 0

def checkVowels(letter):
    for i in range(len(vowels)):
        if letter == vowels[i]:
            return True
    return False

for i in range(len(string)):
    if checkVowels(string[i]):
        vowelscounter = vowelscounter + 1

print(f"\n### {vowelscounter} vowel(s) were found in the string. ###")