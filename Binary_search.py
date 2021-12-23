# It returns location of x in given array arr  
# if present, else returns -1 
def binary_search(arr, l, r, x):
    """
    Binary search algorithm.

    :param arr: The array to be searched.
    :type arr: list of ints or floats (numbers)

    :param l, r : The left and right indices
    of the subarray that is to be searched for x. These are passed as arguments so that binary_search() can use them in recursive calls, without having to
    recompute them itself. They must satisfy l <= r .  In other words, if x is present in the array A[l..r], then l <= r . Also note that we do not assume
    anything about the relative order of elements inside A[l..r]. This means that if there are multiple occurrences of x in A[l..r] , then binary_search()
    will return any one of these occurrences; it won’t necessarily return the first occurrence (which would be a violation of basic definition). If you
    want this behavior instead, you can use linear_search(). However, since binary search runs faster than linear search on average even for this simple
    example where we don’t assume anything about element ordering, I decided it was better not to make any assumptions here and follow a more general
    approach instead. Note also that while searching an
    """
    if l <= r:
        
        mid = (l+r) // 2 #extracting the middle element from the array
        
        # If element is present at the middle itself
        if arr[mid] == x:
            return mid
        
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, l, mid - 1, x)
 
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, r, x)
            
    # If we reach here, then the element was not present 
    return -1


# Main Function
if __name__ == "__main__":
    # User input array
    print("Enter the array with comma separated in which element will be searched")
    arr =[int(x) for x in input().split(',')] #the input array will of int type with each element seperated with a comma due to the split fucntion
                                       #map function returns a list of results after applying the given function to each item
    x = eval(input("Enter the element you want to search in given array"))

    # Function call
    result = binary_search(arr, 0, len(arr) - 1, x)
     
    #printing the output
    if result != -1:
        print("Element is present at index {}".format(result))
    else:
        print("Element is not present in array")
