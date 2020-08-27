'''
  Author : Mohit Kumar
  
  Python program to find triplets in a given  array whose sum is zero 
'''

# function to print triplets with 0 sum 
def find_Triplets_with_zero_sum(arr, num): 
   
    ''' find triplets in a given  array whose sum is zero 
        
        Parameteres : 
            arr : input array 
            num = size of input array
        Output :
            if triplets found return their values 
            else return "No Triplet Found"
    '''
    # bool variable to check if triplet found or not 
    found = False

    # sort array elements 
    arr.sort() 

    # Run a loop until l is less than r, if the sum of array[l], array[r] is equal to zero then print the triplet and break the loop
    for index in range(0, num - 1) : 
    
        # initialize left and right 
        left = index + 1
        right = num - 1

        curr = arr[index] # current element
        
        while (left < right): 
            
            temp = curr + arr[left] + arr[right] 
            
            if (temp == 0) : 
                # print elements if it's sum is zero 
                print(curr, arr[left], arr[right]) 
                
                left += 1
                right -= 1
                
                found = True
            

            # If sum of three elements is less  than zero then increment in left 
            elif (temp < 0) : 
                left += 1

            # if sum is greater than zero than decrement in right side 
            else: 
                right -= 1
        
    if (found == False): 
        print(" No Triplet Found") 

# DRIVER CODE STARTS

if __name__ == "__main__":
    
    n = int(input('Enter size of array\n'))
    print('Enter elements of array\n')
    
    arr = list(map(int,input().split()))
    
    print('Triplets with 0 sum are as : ')
    
    find_Triplets_with_zero_sum(arr, n) 

'''
SAMPLE INPUT 1 :
	Enter size of array : 5 
	Enter elements of array : 0, -1, 2, -3, 1
OUTPUT :
	Triplets with 0 sum are as : 
				    -3 1 2
				    -1 0 1
COMPLEXITY ANALYSIS :
Time Complexity : O(n^2).
    Only two nested loops is required, so the time complexity is O(n^2).
Auxiliary Space : O(1), no extra space is required, so the time complexity is constant.
'''
