from sys import argv  # import argment variable

script, rows, columns = argv  # define rows and columns for the table and assign them to the argument variable


def table(rows, columns):
    for i in range(1, int(
            rows) + 1):  # it's safe to assume that the user would mean 12 rows when they provide 12 as an argument, b'coz 12 will produce 11 rows
        print("\t", i, )

    print("\n\n")  # add 3 lines

    for i in range(1, int(columns) + 1):
        print(i),
        for j in range(1, int(rows) + 1):
            print("\t", i * j, )
        print("\n\n")  # add 3 lines


table(rows, columns)
