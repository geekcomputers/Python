# Update the value of dictionary written by the user...

print("Dictinary opperations")


def Dictionary(Dict, key, value):
    print("Original dictionary", Dict)
    Dict[key] = value
    print("Updated dictionary", Dict)


d = eval(input("Enter the dictionary"))
print("Dictionary", d, "\n")

k = input("Enter the key to be updated")
if k in d.keys():
    v = input("Enter the updated value")
    Dictionary(d, k, v)

else:
    print("Key not found")
