argv = input('Number of ROWS and COLUMBS: ')

script, rows, columns = argv, argv, argv # define rows and columns for the table and assign them to the argument variable


def table(rows, columns):
	for i in range(1, int(columns) + 1 ):
		print (i),
		for j in range(1, int(rows) + 1 ):
			print ("\t",i*j,)
		print ("\n\n") # add 3 lines
		
table(rows, columns)
