
#know what is Pigeonhole_principle
# https://www.youtube.com/watch?v=IeTLZPNIPJQ


def pigeonhole_sort(a): 

	# (number of pigeonholes we need) 
	my_min = min(a) 
	my_max = max(a) 
	size = my_max - my_min + 1

	# total pigeonholes 
	holes = [0] * size 

	# filling up the pigeonholes. 
	for x in a: 	
		holes[x - my_min] += 1

	# Put the elements back into the array in order. 
	i = 0
	for count in range(size): 
		while holes[count] > 0: 
			holes[count] -= 1
			a[i] = count + my_min 
			i += 1
    
 
			

a = [10, 3, 2, 7, 4, 6, 8] 

#list only integers
print(pigeonhole_sort(a) )
print(a)

	
