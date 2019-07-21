# AUTHOR: ekbalba
# DESCRIPTION: A simple script which checks if a given phrase is a Palindrome
# PALINDROME: A word, phrase, or sequence that reads the same backward as forward

samplePhrase = "A man, a plan, a cat, a ham, a yak, a yam, a hat, a canal-Panama!"
#givenPhrase = ""
#phrase = ""

givenPhrase = input("\nPlease input a phrase:(Press ENTER to use the sample phrase) ")

if givenPhrase == "":
    print("\nThe sample phrase is: {0}".format(samplePhrase))
    phrase = samplePhrase
else:
    phrase = givenPhrase

phrase = phrase.lower()


length_ = len(phrase)
bol_ = True

# check using two pointers, one at beginning
# other at the end. Use only half of the list.
for items in range(length_//2):
    if phrase[items] != phrase[length_ - 1 - items]:
        print("\nSorry, The given phrase is not a Palindrome.")
        bol_ = False
        break

if bol_:
    print("\nWow!, The phrase is a Palindrome!")