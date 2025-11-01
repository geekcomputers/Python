# Program to check if a string is palindrome or not

my_str = input().strip()

# make it suitable for caseless comparison
my_str = my_str.casefold()

# reverse the string
rev_str = my_str[::-1]

# check if the string is equal to its reverse
if my_str == rev_str:
    print("The string is a palindrome.")
else:
    print("The string is not a palindrome.")
