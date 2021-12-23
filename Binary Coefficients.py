def pascal_triangle(lineNumber):
    """
    This function returns the list of numbers in a given line number of Pascal's triangle.
    The first line is always 1, and each subsequent line consists
    of the sum of the two numbers above it in the triangle.
    The function takes one argument: an integer that represents which row to return (the first row
    is row 0).
    """
    list1 = list()
    list1.append([1])
    i = 1
    while (i <= lineNumber):
        j = 1
        l = []
        l.append(1)
        while (j < i):
            l.append(list1[i - 1][j] + list1[i - 1][j - 1])
            j = j + 1
        l.append(1)
        list1.append(l)
        i = i + 1
    return list1


def binomial_coef(n, k):
    pascalTriangle = pascal_triangle(n)
    return (pascalTriangle[n][k - 1])
