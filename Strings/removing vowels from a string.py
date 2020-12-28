str=input("Type the string:")
new=""

for ch in str:
    if ch not in "aeiouAEIOU":
        new=new+ch

print(new)