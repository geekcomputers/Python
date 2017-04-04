def countChars(filename):
	count = {}

	with open(inputFile) as info:
		readFile = info.read()
		for character in readFile.upper():
			count[character] = count.get(character, 0) + 1

	return count

if __name__ == '__main__':
	import pprint
	inputFile = input("File Name : ")
	print(countChars(inputFile))
