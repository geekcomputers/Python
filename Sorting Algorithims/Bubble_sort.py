def bubble_sort(Lists):
    for i in range(len(Lists)):
        for j in range(len(Lists) - 1):
            # We check whether the adjecent number is greater or not
            if Lists[j] > Lists[j + 1]:
                Lists[j], Lists[j + 1] = Lists[j + 1], Lists[j]


# Lets the user enter values of an array and verify by himself/herself
array = []
array_length = int(
    input("Enter the number of elements of array or enter the length of array")
)
for i in range(array_length):
    value = int(input("Enter the value in the array"))
    array.append(value)

bubble_sort(array)
print(array)
