# AUTHOR: ekbalba
# DESCRIPTION: A simple script which checks if a given phrase is a Palindrome
# PALINDROME: A word, phrase, or sequence that reads the same backward as forward

samplePhrase = "A man, a plan, a cat, a ham, a yak, a yam, a hat, a canal-Panama!"  
# givenPhrase = ""
# phrase = ""

givenPhrase = input("\nPlease input a phrase:(Press ENTER to use the sample phrase) ") #takes a phrase for input

#if nothing in given as input then the sample phrase is stored in the variable phrase otherwise the given phrase if stored
if givenPhrase == "":
    print("\nThe sample phrase is: {0}".format(samplePhrase))
    phrase = samplePhrase 
else:
    phrase = givenPhrase

phrase = ''.join([c for c in phrase.lower() if c.isalpha() or c.isdigit()]) #converting all the characters of the phrase to the lowercase

length_ = len(phrase) #returns the length of string
bol_ = True

# check using two pointers, one at beginning
# other at the end. Use only half of the list.
for items in range(length_ // 2):
    if phrase[items] != phrase[length_ - 1 - items]:  
        print("\nSorry, The given phrase is not a Palindrome.")
        bol_ = False
        break

if bol_ == True:
    print("\nWow!, The phrase is a Palindrome!")
    
    
    
    
    
    
"""
Method #2:

A simple mmethod is , to reverse the string and and compare with original string.
If both are same that's means string is palindrome otherwise else. 
"""
if phrase==phrase[::-1]:#slicing technique
    """phrase[::-1] this code is for reverse a string very smartly """
    
    print("\nBy Method 2: Wow!, The phrase is a Palindrome!")
else:
    print("\nBy Method 2: Sorry, The given phrase is not a Palindrome.")
