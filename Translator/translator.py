  
from translate import Translator

lang1=input("Enter a language you want to translate from: ")
lang2=input("Enter a language you want to translate to: ")
t=Translator(from_lang = lang1.capitalize(), to_lang=lang2.capitalize())

text=input("Enter text you want to translate: ")
ans=t.translate(text)

print("Translated text is: ")
print(ans)
