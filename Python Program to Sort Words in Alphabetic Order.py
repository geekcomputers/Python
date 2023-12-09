# Program to sort words alphabetically and print them out in a list

#declaring variables
my_str = "Hello this Is an Example With cased letters"
word_Dict = {}
count = 0

#Need to make all words the same case, otherwise, the .sort() function sorts them by ASCII code and they will not appear alphabetically. 
my_str = my_str.lower()

# To take input from the user
#my_str = input("Enter a string: ")

# breakdown the string into a list of words
words = my_str.split()
words.sort()

print(words)
