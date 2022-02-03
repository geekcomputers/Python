list = []

N = int(input("Enter The Size Of List"))

for i in range(0, N):
    a = int(input("Enter The number"))
    list.append(a)


# Let's sort list in ascending order using Selection Sort
# Every time The Element Of List is fetched and the smallest element in remaining list is found and if it comes out
# to be smaller than the element fetched then it is swapped with smallest number.

for i in range(0, len(list) - 1):
    smallest = list[i + 1]
    k = 0
    for j in range(i + 1, len(list)):
        if list[j] <= smallest:
            smallest = list[j]
            k = j

    if smallest < list[i]:
        temp = list[i]
        list[i] = list[k]
        list[k] = temp

print(list)
