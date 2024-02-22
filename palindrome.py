#Program to check if the string entered by the user is a palindrome or not

def checkPalindrome(str):
    
    if str == str[::-1]:#Slicing the string,in reverse order
        print("Yes the string is a palindrome")

    else :
        print("No, the string you entered is not a palindrome")

string=input("Enter the string to check if it is a palindrome")

checkPalindrome(string)

#The end
