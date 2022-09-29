# This program rotates a given string letters by letters
# for example:
# input: "Tie"
# Output: ["ieT", "eTi"]


def rotate(n):
    a = list(n)
    if not a:
        return print([])
    l = []
    for _ in range(1, len(a) + 1):
        a = [a[(i + 1) % (len(a))] for i in range(len(a))]
        l += ["".join(a)]
    print(l)


string = str(input())
print("Your input is :", string)
print("The rotation is :")
rotate(string)


# Input : Python
# output :
# The rotation is :
# ['ythonp', 'thonpy', 'honpyt', 'onpyth', 'npytho', 'python']
