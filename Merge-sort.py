#merge sort

l = []

n = int(input("Enter number of elements in the list: "))

for i in range(n):
    temp = int(input("Enter element"+str(i+1)+': '))
    l += [temp]

def merge_sort(L):

    mid = int( (len(L) - 1)/2 )

    if len(L) > 2:
        a = merge_sort(L[0:mid])
        b = merge_sort(L[mid:len(L)])

    elif len(L) == 2:
        a = [L[0]]
        b = [L[1]]
    else:
        return L
    
    i = 0
    j = 0
    new = []
    while (i!= len(a) or j != len(b)):

        if i < len(a) and j < len(b):

            if a[i] < b[j]:
                new += [a[i]]
                i += 1
                
            else:
                new +=[b[j]]
                j += 1

        elif i == len(a) and j != len(b):
            new += [b[j]]
            j += 1
            
        elif j == len(b) and i != len(a):
            new += [a[i]]
            i += 1

        else:
            break

    return new

print(merge_sort(l))
