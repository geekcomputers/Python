"""

A simple method is , to reverse the string and and compare with original string.
If both are same that's means string is palindrome otherwise else. 
"""
phrase = input()
if phrase == phrase[::-1]:  # slicing technique
    """phrase[::-1] this code is for reverse a string very smartly """

    print("\n Wow!, The phrase is a Palindrome!")
else:
    print("\n Sorry, The given phrase is not a Palindrome.")
