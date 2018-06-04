# AUTHOR: ekbalba
# DESCRIPTION: A simple script which checks if a given phrase is a Palindrome
# PALINDROME: A word, phrase, or sequence that reads the same backward as forward

samplePhrase = "A man, a plan, a cat, a ham, a yak, a yam, a hat, a canal-Panama!"
#givenPhrase = ""
#phrase = ""

givenPhrase = input("\nPlease input a phrase:(Press ENTER to use the sample phrase) ")

if givenPhrase == "" or not givenPhrase.strip():
    print("\nThe sample phrase is: {0}".format(samplePhrase))
    phrase = samplePhrase
else:
    phrase = givenPhrase

string = phrase.lower()

if string == string[::-1]:
    print("\nWow!, The phrase is a Palindrome!")
else:
    print("\nSorry, The given phrase is not a Palindrome.")
