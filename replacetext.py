#!/usr/bin/env python3
# program to replace all the spaces in an entered string with a hyphen"-"
def replacetext(string):
    string = string.replace(" ", "-")
    return string


S = input("Enter a text to replace all its spaces with hyphens: ")
N = replacetext(S)
print("The changed text is: ", N)
