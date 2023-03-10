def vowel_remover(text):
    string = ""
    for l in text:
        if l.lower() not in "aeiou":
            string += l
    return string
print(vowel_remover("hello world!"))
