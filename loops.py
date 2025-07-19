# 2 loops

# for loop:

"""
Syntax..
-> "range" : starts with 0.
-> The space after the space is called as identiation, python generally identifies the block of code with the help of indentation,
indentation is generally 4 spaces / 1 tab space..


for <variable> in range(<enter the range>):
    statements you want to execute

for <varaible> in <list name>:
    print(<variable>)
To print the list / or any iterator items

"""

# 1. for with range...
for i in range(3):
    print("Hello... with range")
    # prints Hello 3 times..

# 2.for with list

l1 = [1, 2, 3, 78, 98, 56, 52]
for i in l1:
    print("list items", i)
    # prints list items one by one....

for i in "ABC":
    print(i)

# while loop:
i = 0
while i <= 5:
    print("hello.. with while")
    i += 1
