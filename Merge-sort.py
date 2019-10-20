# merge sort

l = []  # declaring list l

n = int(input("Enter number of elements in the list: "))  # taking value from user

for i in range(n):
    temp = int(input("Enter element" + str(i + 1) + ': '))
    l += [temp]


def merge_sort(L):
    mid = int((len(L) - 1) / 2)  # calculate the value of middle index of array

    if len(L) > mid + 2:  # this loop will run when length of array is greater than 2
        a = merge_sort(L[0:mid])  # a is initialised with left side of the array taking reference as middle index
        b = merge_sort(L[mid:len(L)])  # b is initialised with right side of the array taking reference as middle index

    elif len(L) == 2:  # this loop will when length of array is equal to 2
        a = [L[0]]  # a is initiliased with the first element of array
        b = [L[1]]  # b is initiliased with the second element of array
    else:
        return L

    i = 0
    j = 0
    new = []
    while (i != len(a) or j != len(
            b)):  # this loop will run until i is not equal to the lenth of array a or j is equal to length of array b

        if i < len(a) and j < len(
                b):  # checking if value of i and j is lesser than length of array a and b respectively

            if a[i] < b[j]:  # if the element on the left side is lesser than the element on the right side
                new += [a[i]]  # then it will be directly added to the array new
                i += 1  # i is increased by 1

            else:
                new += [b[
                            j]]  # if element on the right side is lesser than the element on the left side then right side element is added to the array new
                j += 1  # j increased by 1

        elif i == len(a) and j != len(
                b):  # if i gets equal to the length of array a then all the elements of array b are directly added to the array new
            new += [b[j]]
            j += 1

        elif j == len(b) and i != len(
                a):  # if j gets equal to the length of array b then all the elements of array a are directly added to the array new
            new += [a[i]]
            i += 1

        else:
            break

    return new


print(merge_sort(l))
