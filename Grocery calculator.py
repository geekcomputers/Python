'''This will be a Python script that functions as a grocery calculator. It will take in key-value pairs for items
and their prices, and return the subtotal and total, and can print out the list for you for when you're ready to
take it to the store!'''

'''Algorithm:
1. User enters key-value pairs that are added into a dict.
2. Users tells script to return total, subtotal, and key-value pairs in a nicely formatted list.'''

#Object = GroceryList
#Methods = addToList, Total, Subtotal, returnList
class GroceryList(dict):

	def __init__(self):
		self = {}

	def addToList(self, item, price):
        
		self.update({item:price})

	def Total(self):
		total = 0
		for items in self:
			total += (self[items])*.07 + (self[items])
		return total

	def Subtotal(self):
		subtotal = 0
		for items in self:
			subtotal += self[items]
		return subtotal

	def returnList(self):
		return self

'''Test list should return:
Total = 10.70
Subtotal = 10
returnList = {"milk":4, "eggs":3, "kombucha":3}
'''
List1 = GroceryList()

List1.addToList("milk",4)
List1.addToList("eggs", 3)
List1.addToList("kombucha", 3)


print(List1.Total())
print(List1.Subtotal())
print(List1.returnList())

#*****************************************************
print()
#*****************************************************


List2 = GroceryList()

List2.addToList('cheese', 7.49)
List2.addToList('wine', 25.36)
List2.addToList('steak', 17.64)

print(List2.Total())
print(List2.Subtotal())
print(List2.returnList())
