# program to find a missing number from a list.


def missing_number(num_list):
    return sum(range(num_list[0], num_list[-1] + 1)) - sum(num_list)


print(missing_number([1, 2, 3, 4, 6, 7, 8]))

print(missing_number([10, 11, 12, 14, 15, 16, 17]))
