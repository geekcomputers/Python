#!/usr/bin/env python3

"""
Python Program using the Luhn Algorithm

This program uses the Luhn Algorithm, named after its creator
Hans Peter Luhn, to calculate the check digit of a 10-digit
"payload" number, and output the final 11-digit number.

To prove this program correctly calculates the check digit,
the input 7992739871 should return:

Sum of all digits: 67
Check digit: 3
Full valid number (11 digits): 79927398713

11/15/2021
David Costell (DontEatThemCookies on GitHub)
"""

# Input
CC = input("Enter number to validate (e.g. 7992739871): ")
if len(CC) < 10 or len(CC) > 10:
    input("Number must be 10 digits! ")
    exit()

# Use list comprehension to split the number into individual digits
split = [int(split) for split in str(CC)]

# List of digits to be multiplied by 2 (to be doubled)
tobedoubled = [split[1], split[3], split[5], split[7], split[9]]
# List of remaining digits not to be multiplied
remaining = [split[0], split[2], split[4], split[6], split[8]]

# Step 1
# Double all values in the tobedoubled list
# Put the newly-doubled values in a new list
newdoubled = []
for i in tobedoubled:
    i = i * 2
    newdoubled.append(i)
tobedoubled = newdoubled

# Check for any double-digit items in the tobedoubled list
# Splits all double-digit items into two single-digit items
newdoubled = []
for i in tobedoubled:
    if i > 9:
        splitdigit = str(i)
        for index in range(0, len(splitdigit), 1):
            newdoubled.append(splitdigit[index : index + 1])
        tobedoubled.remove(i)
newdoubled = [int(i) for i in newdoubled]

# Unify all lists into one (luhnsum)
luhnsum = []
luhnsum.extend(tobedoubled)
luhnsum.extend(newdoubled)
luhnsum.extend(remaining)

# Output
print("Final digit list:", luhnsum)
print("Sum of all digits:", sum(luhnsum))
checkdigit = 10 - sum(luhnsum) % 10
print("Check digit:", checkdigit)
finalcc = str(CC) + str(checkdigit)
print("Full valid number (11 digits):", finalcc)
input()
