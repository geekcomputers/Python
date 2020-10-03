

# See what stooge sort dooes
# https://www.youtube.com/watch?v=vIDkfrSdID8

def stooge_sort_(arr, l, h): 
	if l >= h: 
		return 0

	# If first element is smaller than last, then swap 

	if arr[l]>arr[h]: 
		t = arr[l] 
		arr[l] = arr[h] 
		arr[h] = t 

	# If there are more than 2 elements in array 
	if h-l + 1 > 2: 
		t = (int)((h-l + 1)/3) 

		# Recursively sort first 2 / 3 elements 
		stooge_sort_(arr, l, (h-t)) 

		# Recursively sort last 2 / 3 elements 
		stooge_sort_(arr, l + t, (h)) 

		# Recursively sort first 2 / 3 elements 
		stooge_sort_(arr, l, (h-t)) 
    
 



arr = [2, 4, 5, 3, 1] 
n = len(arr) 

stooge_sort_(arr, 0, n-1) 

print(arr)