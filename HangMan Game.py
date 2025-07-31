# Program for HangMan Game.
import random, HangMan_Includes as incl

while True:
	chances=6
	inp_lst=[]
	result_lst=[]
	name=random.choice(incl.names).upper()
	# print(name)
	[result_lst.append('__ ') for i in range(len(name))]
	result_str=str().join(result_lst)

	print(f'\nYou have to Guess a Human Name of {len(name)} Alphabets:\t{result_str}')
	print(incl.draw[0])

	while True:
		if result_str.replace(' ','')==name:
			print(f'\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Correct Answer: {name} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			print(incl.won+'\a')
			break
		inp=input('\nGuess an Alphabet or a Sequence of Alphabets: ').upper()
		
		if inp in inp_lst:
			print('......................................................................Already Tried')
			continue
		else:
			inp_lst.append(inp)
		
		t=0
		indx=[]
		if inp in name:
			temp=name
			while temp!='':
				if inp in temp:
					indx.append(t+temp.index(inp))
					t=temp.index(inp)+1
					temp=temp[t:]
				else:
					break
			
			for j in range(len(indx)):
				for i in range(len(inp)):
					result_lst[indx[j]]=inp[i]+' '
					indx[j]+=1
					i+=1
			
			result_str=str().join(result_lst)
			print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Excellent~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			print(f'\nYou have to Guess a Human Name of {len(name)} Alphabets:\t{result_str}\n')
			print('Tried Inputs:',tuple(sorted(set(inp_lst))))

		else:
			print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Try Again!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			print(f'\nYou have to Guess a Human Name of {len(name)} Alphabets:\t{result_str}\n')
			print(incl.draw[chances])
			chances=chances-1
			
			if chances!=0:
				print('Tried Inputs:',tuple(sorted(set(inp_lst))))
				print(f'\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~You were left with {chances} Chances~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			else:
				print(f'\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Correct Answer: {name} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
				print(incl.lose+'\a')
				break
	
	try:
		if int(input('To play the Game Again Press "1" & "0" to Quit: '))!=1:
			exit('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Thank You~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	except:
		exit('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Thank You~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
