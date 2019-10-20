list = []
num = int(input("Enter size of list: \t"))
for n in range(num):
    numbers = int(input("Enter any number: \t"))
    list.append(numbers)

x = int(input("\nEnter number to search: \t"))

found = False

for i in range(len(list)):
    if list[i] == x:
        found = True
        print("\n%d found at position %d" % (x, i))
        break
if not found:
    print("\n%d is not in list" % x)
