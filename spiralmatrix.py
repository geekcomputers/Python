n = int(input("Enter the size of matrix:"))
t = 1
r = 0  # r stands for row
c = 0  # c stands for column
matrix = [[0 for x in range(n)] for y in range(n)]  # to initialise the matrix
if n % 2 == 0:
    k = n // 2
else:
    k = int((n / 2) + 1)
for i in range(k):
    while c < n:
        matrix[r][c] = t
        t = t + 1
        c = c + 1
    r = r + 1
    c = c - 1
    while r < n:
        matrix[r][c] = t
        t = t + 1
        r = r + 1
    r = r - 1
    c = c - 1
    while c >= i:
        matrix[r][c] = t
        c = c - 1
        t = t + 1
    c = c + 1
    r = r - 1
    while r > i:
        matrix[r][c] = t
        t = t + 1
        r = r - 1
    r = r + 1
    n = n - 1
    c = c + 1
for m in matrix:
    print(m)
