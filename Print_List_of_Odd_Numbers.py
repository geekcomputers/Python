# master
# Another best method to do this

n = map(list(int, input().split()))
odd_list = list(i for i in n if i % 2 != 0)
print(odd_list)
exit()

# CALCULATE NUMBER OF ODD NUMBERS

# CALCULATE NUMBER OF ODD NUMBERS WITHIN A GIVEN LIMIT
# master

n = int(input("Enter the limit : "))  # user input

if n <= 0:
    print("Invalid number, please enter a number greater than zero!")
else:
    odd_list = [i for i in range(1, n + 1, 2)]  # creating string with number "i"
    print(odd_list)  # in range from 1 till "n".


#     printing odd and even number in same program
n = map(list(int, input().split()))
even = []
odd = []
for i in range(n):
    if i % 2 == 0:
        even.append(i)
    else:
        odd.append(i)
