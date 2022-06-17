def merge(left_list, right_list):
    sorted_list = []
    left_list_index = right_list_index = 0

    # We use the list lengths often, so its handy to make variables
    left_list_length, right_list_length = len(left_list), len(right_list)

    for _ in range(left_list_length + right_list_length):
        if (
            left_list_index < left_list_length
            and right_list_index < right_list_length
            and left_list[left_list_index] <= right_list[right_list_index]
            or (
                left_list_index >= left_list_length
                or right_list_index >= right_list_length
            )
            and left_list_index != left_list_length
            and right_list_index == right_list_length
        ):
            sorted_list.append(left_list[left_list_index])
            left_list_index += 1
        elif (
            left_list_index < left_list_length
            and right_list_index < right_list_length
            or left_list_index == left_list_length
        ):
            sorted_list.append(right_list[right_list_index])
            right_list_index += 1

    return sorted_list


def merge_sort(nums):
    # If the list is a single element, return it
    if len(nums) <= 1:
        return nums

    # Use floor division to get midpoint, indices must be integers
    mid = len(nums) // 2

    # Sort and merge each half
    left_list = merge_sort(nums[:mid])
    right_list = merge_sort(nums[mid:])

    # Merge the sorted lists into a new one
    return merge(left_list, right_list)


# Verify it works
random_list_of_nums = [120, 45, 68, 250, 176]
random_list_of_nums = merge_sort(random_list_of_nums)
print(random_list_of_nums)

"""
Here merge_sort() function, unlike the previous sorting algorithms, returns a new list that is sorted, rather than sorting the existing list.
Therefore, Merge Sort requires space to create a new list of the same size as the input list
"""
