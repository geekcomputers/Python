# Program to sort words alphabetically and put them in a dictionary with corresponding numbered keys
# We are also removing punctuation to ensure the desired output, without importing a library for assistance. 

# Declare base variables
word_Dict = {}
count = 0
my_str = "Hello this Is an Example With cased letters. Hello, this is a good string"
#Initialize punctuation
punctuations = '''!()-[]{};:'",<>./?@#$%^&*_~'''

# To take input from the user
#my_str = input("Enter a string: ")

# remove punctuation from the string and use an empty variable to put the alphabetic characters into
no_punct = ""
for char in my_str:
   if char not in punctuations:
       no_punct = no_punct + char

# Make all words in string lowercase. my_str now equals the original string without the punctuation 
my_str = no_punct.lower()

# breakdown the string into a list of words
words = my_str.split()

# sort the list and remove duplicate words
words.sort()

new_Word_List = []
for word in words:
    if word not in new_Word_List:
        new_Word_List.append(word)
    else:
        continue

# insert sorted words into dictionary with key

for word in new_Word_List:
    count+=1
    word_Dict[count] = word

print(word_Dict)
