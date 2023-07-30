num = int(input("Enter size of list: \t"))
list = [int(input("Enter any number: \t")) for _ in range(num)]

x = int(input("\nEnter number to search: \t"))

for position, number in enumerate(list):
    if number == x:
        print(f"\n{x} found at position {position}")
else:
    print(f"list: {list}")
    print(f"{x} is not in list")