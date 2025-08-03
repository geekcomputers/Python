#

# With slicing -> Reverses the string using string[::-1]


string = input("enter a word to check.. ")
copy = string[::-1]
if string == copy:
    print("Plaindrome")
else:
    print("!")

# Without slicing –> Reverses the string manually using a loop
reverse_string = ""
for i in string:
    reverse_string = i + reverse_string
if string == reverse_string:
    print(reverse_string)
else:
    print("!")
