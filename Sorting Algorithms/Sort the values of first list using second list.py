# one list using
# the other list


def sort_list(list1, list2):

    zipped_pairs = zip(list2, list1)

    z = [x for _, x in sorted(zipped_pairs)]

    return z


# driver code
x = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
y = [0, 1, 1, 0, 1, 2, 2, 0, 1]

print(sort_list(x, y))

x = ["g", "e", "e", "k", "s", "f", "o", "r", "g", "e", "e", "k", "s"]
y = [0, 1, 1, 0, 1, 2, 2, 0, 1]

print(sort_list(x, y))
