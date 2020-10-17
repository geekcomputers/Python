num=int(input("enter any Number"))
rev =0
while num>0 :
    Rem = num% 10
    num = num//10
    rev=rev*10+Rem
print("The Reverse of the number",rev)
