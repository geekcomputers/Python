List = []
# List is Muteable
# means value can be change
List.insert(0, 5) #insertion takes place at mentioned index
List.insert(1, 10) 
List.insert(0, 6)
print(List)
List.remove(6) 
List.append(9) #insertion takes place at last 
List.append(1)
List.sort()    #arranges element in ascending order
print(List)
List.pop()
List.reverse()
print(List)
"""
List.append(1)
print(List)
List.append(2)
print(List)
List.insert(1 , 3)
print(List)
"""

list2 = [2, 3, 7, 5, 10, 17, 12, 4, 1, 13]
for i in list2:
    if i % 2 == 0:
        print(i)
"""
Expected Output:
2
10
12
4
"""
