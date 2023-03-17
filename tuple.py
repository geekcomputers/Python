a=(1,2,3,4,5)
print('Individual elements of Tuple are')
for i in a:
    print(i,end='  ')
print()
print('Value at index number 3 is:',a[3])
print('Values from index no 2 are',a[2:])
print('Length of tuple is',len(a))
print('Maximum value from tuple is ',max(a))
print('Minimum value from tuple is ',min(a))
del a    #delete the whole tuple
