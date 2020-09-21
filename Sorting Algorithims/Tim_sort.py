'''   Author : Mohit Kumar
      
        Tim Sort implemented in python
        Time Complexity : O(n log(n))
        Space Complexity :O(n)

'''

# Python3 program to perform TimSort.  
RUN = 32 
    
# This function sorts array from left index to  
# to right index which is of size atmost RUN  
def insertionSort(arr, left, right):  
   
    for i in range(left + 1, right+1):  
       
        temp = arr[i]  
        j = i - 1 
        while j >= left and arr[j] > temp :  
           
            arr[j+1] = arr[j]  
            j -= 1
           
        arr[j+1] = temp  
    
# merge function merges the sorted runs  
def merge(arr, l, m, r): 
   
    # original array is broken in two parts  
    # left and right array  
    len1, len2 =  m - l + 1, r - m  
    left, right = [], []  
    for i in range(0, len1):  
        left.append(arr[l + i])  
    for i in range(0, len2):  
        right.append(arr[m + 1 + i])  
    
    i, j, k = 0, 0, l 
    # after comparing, we merge those two array  
    # in larger sub array  
    while i < len1 and j < len2:  
       
        if left[i] <= right[j]:  
            arr[k] = left[i]  
            i += 1 
           
        else: 
            arr[k] = right[j]  
            j += 1 
           
        k += 1
       
    # copy remaining elements of left, if any  
    while i < len1:  
       
        arr[k] = left[i]  
        k += 1 
        i += 1
    
    # copy remaining element of right, if any  
    while j < len2:  
        arr[k] = right[j]  
        k += 1
        j += 1
      
# iterative Timsort function to sort the  
# array[0...n-1] (similar to merge sort)  
def timSort(arr, n):  
   
    # Sort individual subarrays of size RUN  
    for i in range(0, n, RUN):  
        insertionSort(arr, i, min((i+31), (n-1)))  
    
    # start merging from size RUN (or 32). It will merge  
    # to form size 64, then 128, 256 and so on ....  
    size = RUN 
    while size < n:  
       
        # pick starting point of left sub array. We  
        # are going to merge arr[left..left+size-1]  
        # and arr[left+size, left+2*size-1]  
        # After every merge, we increase left by 2*size  
        for left in range(0, n, 2*size):  
           
            # find ending point of left sub array  
            # mid+1 is starting point of right sub array  
            mid = left + size - 1 
            right = min((left + 2*size - 1), (n-1))  
    
            # merge sub array arr[left.....mid] &  
            # arr[mid+1....right]  
            merge(arr, left, mid, right)  
          
        size = 2*size 
           
# utility function to print the Array  
def printArray(arr, n):  
   
    for i in range(0, n):  
        print(arr[i], end = " ")  
    print()  
   
if __name__ == "__main__": 
    
    n = int(input('Enter size of array\n'))
    print('Enter elements of array\n')
    
    arr = list(map(int ,input().split()))  
    print("Given Array is")  
    printArray(arr, n)  
    
    timSort(arr, n)  
    
    print("After Sorting Array is")  
    printArray(arr, n)  

''' 
    OUTPUT : 
    
    Enter size of array : 5
    Given Array is
    5 3 4 2 1 
    After Sorting Array is
    1 2 3 4 5
      
'''
   
