import random

lChars = "abcdefghijklmnopqrstuvwxyz"
uChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
digits = "1234567890"
specialChars = "!@#$%^&*-_+="

myPass = ""

# Generate 3 lowercase letters
for _ in range(3):
    myPass += random.choice(lChars)

# Generate 3 digits
for _ in range(3):
    myPass += random.choice(digits)

# Generate 2 special characters
for _ in range(2):
    myPass += random.choice(specialChars)

# Generate 2 uppercase letters
for _ in range(2):
    myPass += random.choice(uChars)

print(myPass)  # Output: 10-character password (e.g. "abc123!@AB")
