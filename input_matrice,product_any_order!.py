# inputing 2 matrices:

# matrice 1:

rows = int(input("Enter the number of rows of the matrice 1"))
coloumns = int(input("Enter the coloumns of the matrice 1"))
matrice = []
rowan = []

for i in range(0, rows):
    for j in range(0, coloumns):
        element = int(input("enter the element"))
        rowan.append(element)
    print("one row completed")
    matrice.append(rowan)
    rowan = []

print("matrice 1 is \n")
for ch in matrice:
    print(ch)
A = matrice

# matrice 2:

rows_ = coloumns
coloumns_ = int(input("Enter the coloumns of the matrice 2"))
rowan = []
matrix = []

for i in range(0, rows_):
    for j in range(0, coloumns_):
        element = int(input("enter the element"))
        rowan.append(element)
    print("one row completed")
    matrix.append(rowan)
    rowan = []

print("Matrice 2 is\n")
for ch in matrix:
    print(ch)

B = matrix

# creating empty frame:

result = []
for i in range(0, rows):
    for j in range(0, coloumns_):
        rowan.append(0)
    result.append(rowan)
    rowan = []
print("\n")
print("The frame work of result")
for ch in result:
    print(ch)


# Multiplication of the two matrices:

for i in range(len(A)):
    for j in range(len(B[0])):
        for k in range(len(B)):
            result[i][j] += A[i][k] * B[k][j]

print("\n")
print("The product of the 2 matrices is \n")

for i in result:
    print(i)
