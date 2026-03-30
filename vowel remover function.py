def vowel_remover(text):
    string = ""
    for l in text:
        if l.lower() not in "aeiou":
            string += l
    return string

# this code runes on only this file
if __name__=="__main__":
    print(vowel_remover("hello world!"))
