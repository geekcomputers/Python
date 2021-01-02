def bubble_sort(nums):
    for i in range(len(nums)):
        for j in range(len(nums)-1):
            # We check whether the adjecent number is greater or not
            if nums[j]>nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]

#Lets the user enter values of an array and verify by himself/herself
array = []
array_length = int(input(print("Enter the number of elements of array or enter the length of array")))
for i in range(array_length):
    value = int(input(print("Enter the value in the array")))
    array.append(value)
    
bubble_sort(array)    
print(array)
