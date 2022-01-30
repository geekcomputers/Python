vowels = "aeiou"

ip_str = "Hello, have you tried our tutorial section yet?"


# count the vowels
vowel_count = 0
consonant_count = 0

for char in ip_str:
    if char in vowels:
        vowel_count += 1
    else:
        consonant_count += 1

print("Total Vowels: ", vowel_count)
print("Total consonants: ", consonant_count)
