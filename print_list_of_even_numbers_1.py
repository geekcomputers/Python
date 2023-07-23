# Very sort method to creat list of even number form a given list
# Advance-Python
list_number = list(map(int, input().split()))
even_list = [i for i in list_number if i % 2 == 0]
print(even_list)
exit()  # Another one
n = int(input("Enter the required range : "))  # user input
list = []

if n < 0:
    print("Not a valid number, please enter a positive number!")
else:
    for i in range(0, n + 1):
        if i % 2 == 0:
            list.append(
                i
            )  # appending items to the initialised list getting from the 'if' statement

print(list)
