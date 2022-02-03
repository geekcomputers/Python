#!/usr/bin/python3

# find phone numbers and email addresses
# ./ph_email.py searches for phone numbers and emails in the latest clipboard
# entry and writes the matches into matches.txt

import re

import pyperclip

# Phone regex overview per line
# word boundary
# area code +91, 91, 0
# optional space
# ten numbers
# word boundary

find_phone = re.compile(
    r"""\b
							(\+?91|0)?
							\ ?
							(\d{10})
							\b
							""",
    re.X,
)

# email regex source : http://www.regexlib.com/REDetails.aspx?regexp_id=26
find_email = re.compile(
    r"""(
							([a-zA-Z0-9_\-\.]+)	
							@
							((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)
							|
							(([a-zA-Z0-9\-]+\.)+))
							([a-zA-Z]{2,4}|[0-9]{1,3})
							(\]?)
							)
							""",
    re.X,
)

text = pyperclip.paste()  # retrieve text from clipboard

matches = []  # list to store numbers and emails

# ph[1] means second item of the group-wise tuple
# which is returned by findall function
# same applies to email

for ph in find_phone.findall(text):
    matches.append(ph[1])

for em in find_email.findall(text):
    matches.append(em[0])

# display number of matches
print(f"{len(matches)} matches found")

# if matches are found add then to file
if len(matches):
    with open("matches.txt", "a") as file:
        for match in matches:
            file.write(match)
            file.write("\n")
