def pascal_triangle(lineNumber):
    list1 = [[1]]
    i = 1
    while i <= lineNumber:
        l = [1]
        l.extend(list1[i - 1][j] + list1[i - 1][j - 1] for j in range(1, i))
        l.append(1)
        list1.append(l)
        i += 1
    return list1


def binomial_coef(n, k):
    pascalTriangle = pascal_triangle(n)
    return pascalTriangle[n][k - 1]
