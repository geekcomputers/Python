s=input()

'''

s is the input which is to be capitalized

'''

l1=s.split()
print(l1)
l2 = [x.capitalize() for x in l1]

'''

l2 list contains capitalized words of s input

'''

print(l2)
x=" "
print(x.join(l2))
