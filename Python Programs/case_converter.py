s = input("Enter a string: ")
new_s = ""
for c in s:
    if 'A' <= c <= 'Z':
        new_s += chr(ord(c) + 32)
    elif 'a' <= c <= 'z':
        new_s += chr(ord(c) - 32)
    else:
        new_s += c
print("Converted string:", new_s)
