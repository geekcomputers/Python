# GGearing
# 02/10/2017
# Simple script to calculate the quadratic formula of a sequence of numbers and
# recognises when the sequence isn't quadratic


def findLinear(numbers):  # find a & b of linear sequence
    a = numbers[1] - numbers[0]
    a1 = numbers[2] - numbers[1]
    if a1 == a:
        b = numbers[0] - a
        return (a, b)
    else:
        print("Sequence is not linear")


sequence = []
first_difference = []
second_difference = []
for i in range(4):  # input
    term = str(i + 1)
    inp = int(input("Enter term " + term + ": "))
    sequence.append(inp)

for i in range(3):
    gradient = sequence[i + 1] - sequence[i]
    first_difference.append(gradient)
for i in range(2):
    gradient = first_difference[i + 1] - first_difference[i]
    second_difference.append(gradient)

if second_difference[0] == second_difference[1]:  # checks to see if consistent
    a = second_difference[0] / 2
    subs_diff = []
    for i in range(4):
        n = i + 1
        num = a * (n * n)
        subs_diff.append((sequence[i]) - num)
    b, c = findLinear(subs_diff)
    print(
        "Nth term: " + str(a) + "n^2 + " + str(b) + "n + " + str(c)
    )  # outputs nth term
else:
    print("Sequence is not quadratic")
