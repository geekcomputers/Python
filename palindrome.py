def is_palindrome(text):
    text = text.lower()

    cleaned = ""
    for char in text:
        if char.isalnum():
            cleaned += char

    reversed_text = cleaned[::-1]
    return cleaned == reversed_text


user_input = input("Enter a word or a sentence:")
if is_palindrome(user_input):
    print("It's a palindrome")
else:
    print("It's not a palindrome")
