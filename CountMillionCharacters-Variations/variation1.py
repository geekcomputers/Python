import pprint

inputFile = input('File Name: ') 

count = { }
with open(inputFile, 'r') as info:
	readFile = info.read()
	for character in readFile.upper():
		count.setdefault(character, 0)
		count[character] = count[character]+1
		
value = pprint.pformat(count)
print(value)
