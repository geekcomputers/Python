# counting sort

l = []

n = int(input("Enter number of elements in the list: "))

highest = 0

for i in range(n):
    temp = int(input("Enter element" + str(i + 1) + ': '))

    if temp > highest:
        highest = temp

    l += [temp]


def counting_sort(l, h):
    bookkeeping = [0 for i in range(h + 1)]

    for i in l:
        bookkeeping[i] += 1

    L = []

    for i in range(len(bookkeeping)):

        if bookkeeping[i] > 0:

            for j in range(bookkeeping[i]):
                L += [i]

    return L


print(counting_sort(l, highest))
