# merge sort

lst = []  # declaring list l

n = int(input("Enter number of elements in the list: "))  # taking value from user

for i in range(n):
    temp = int(input("Enter element" + str(i + 1) + ': '))
    lst.append( temp )


def merge( ori_lst, left, mid, right ):
	L, R = [], []					# PREPARE TWO TEMPORARY LIST TO HOLD ELEMENTS
	for i in range( left, mid ):	# LOADING
		L.append( ori_lst[i] )
	for i in range( mid, right ):	# LOADING
		R.append( ori_lst[i] )
	base = left						# FILL ELEMENTS BACK TO ORIGINAL LIST START FROM INDEX LEFT
	# EVERY LOOP CHOOSE A SMALLER ELEMENT FROM EITHER LIST
	while len(L) > 0 and len(R) > 0:
		if L[0] < R[0]:
			ori_lst[base] = L[0]
			L.remove( L[0] )
		else:
			ori_lst[base] = R[0]
			R.remove( R[0] )
		base += 1
	# UNLOAD THE REMAINER
	while len( L ) > 0:				
		ori_lst[base] = L[0]
		L.remove( L[0] )
		base += 1
	while len( R ) > 0:
		ori_lst[base] = R[0]
		R.remove( R[0] )
		base += 1
	# ORIGINAL LIST SHOULD BE SORTED FROM INDEX LEFT TO INDEX RIGHT


def merge_sort( L, left, right ):
	if left+1 >= right: 			# ESCAPE CONDITION
		return
	mid = left + ( right - left ) // 2
	merge_sort( L, left, mid ) 		# LEFT
	merge_sort( L, mid, right ) 	# RIGHT
	merge( L, left, mid, right ) 	# MERGE


print( "UNSORTED -> ", lst )
merge_sort( lst, 0, n )
print( "SORTED -> ", lst )
