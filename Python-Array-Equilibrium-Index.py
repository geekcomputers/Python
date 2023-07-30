"""Array Equilibrium Index
Send Feedback
Find and return the equilibrium index of an array. Equilibrium index of an array is an index i such that the sum of elements at indices less than i is equal to the sum of elements at indices greater than i.
Element at index i is not included in either part.
If more than one equilibrium index is present, you need to return the first one. And return -1 if no equilibrium index is present.
Input format :
Line 1 : Size of input array
Line 2 : Array elements (separated by space)
Constraints:
Time Limit: 1 second
Size of input array lies in the range: [1, 1000000]
Sample Input :
7
-7 1 5 2 -4 3 0
Sample Output :
3 """
def equilibrium(arr): 
  
    # finding the sum of whole array 
    total_sum = sum(arr) 
    leftsum = 0
    for i, num in enumerate(arr): 
          
        # total_sum is now right sum 
        # for index i 
        total_sum -= num 
          
        if leftsum == total_sum: 
            return i 
        leftsum += num 
       
      # If no equilibrium index found,  
      # then return -1 
    return -1
n = int(input())
arr = [int(i) for i in input().strip().split()]
print(equilibrium(arr))
