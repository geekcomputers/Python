str=input("Type the string:")
str1=""
for i in str:
    if i in "aeiouAEIOU":
        i="*"
    str1=str1+i

print(str1)