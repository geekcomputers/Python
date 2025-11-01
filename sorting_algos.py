"""Contains some of the Major Sorting Algorithm"""


def selection_sort(arr: list) -> list:
    """Sorts a list in ascending order using the selection sort algorithm.

    Args:
        arr: List of comparable elements (e.g., integers, strings).

    Returns:
        The sorted list (in-place modification).

    Examples:
    >>> selection_sort([])
    []
    >>> selection_sort([1])
    [1]
    >>> selection_sort([1, 2, 3, 4])
    [1, 2, 3, 4]
    >>> selection_sort([4, 3, 2, 1])
    [1, 2, 3, 4]
    >>> selection_sort([3, 1, 3, 2])
    [1, 2, 3, 3]
    >>> selection_sort([5, 5, 5, 5])
    [5, 5, 5, 5]
    >>> selection_sort([2, 4, 1, 3])
    [1, 2, 3, 4]
    >>> selection_sort([-1, -3, 0, 2])
    [-3, -1, 0, 2]
    >>> selection_sort([0, -5, 3, -2, 1])
    [-5, -2, 0, 1, 3]
    >>> selection_sort(["banana", "apple", "cherry"])
    ['apple', 'banana', 'cherry']
    >>> selection_sort(["Apple", "banana", "Cherry"])
    ['Apple', 'Cherry', 'banana']
    >>> selection_sort([2147483647, -2147483648, 0])
    [-2147483648, 0, 2147483647]
    """

    """TC : O(n^2)
    SC : O(1)"""

    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr


def bubble_sort(arr: list) -> list:
    """TC : O(n^2)
    SC : O(1)"""
    n = len(arr)
    flag = True
    while flag:
        flag = False
        for i in range(1, n):
            if arr[i - 1] > arr[i]:
                flag = True
                arr[i - 1], arr[i] = arr[i], arr[i - 1]
    return arr


def insertion_sort(arr: list) -> list:
    """TC : O(n^2)
    SC : O(1)"""
    n = len(arr)
    for i in range(1, n):
        for j in range(i, 0, -1):
            if arr[j - 1] > arr[j]:
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
            else:
                break
    return arr


def merge_sort(arr: list) -> list:
    """TC : O(nlogn)
    SC : O(n) for this version ... But SC can be reduced to O(1)"""
    n = len(arr)
    if n == 1:
        return arr

    m = len(arr) // 2
    L = arr[:m]
    R = arr[m:]
    L = merge_sort(L)
    R = merge_sort(R)
    l = r = 0

    sorted_arr = [0] * n
    i = 0

    while l < len(L) and r < len(R):
        if L[l] < R[r]:
            sorted_arr[i] = L[l]
            l += 1
        else:
            sorted_arr[i] = R[r]
            r += 1
        i += 1

    while l < len(L):
        sorted_arr[i] = L[l]
        l += 1
        i += 1

    while r < len(R):
        sorted_arr[i] = R[r]
        r += 1
        i += 1

    return arr


def quick_sort(arr: list) -> list:
    """TC : O(nlogn) (TC can be n^2 for SUUUper worst case i.e. If the Pivot is continuously bad)
    SC : O(n) for this version ... But SC can be reduced to O(logn)"""

    if len(arr) <= 1:
        return arr

    piv = arr[-1]
    L = [x for x in arr[:-1] if x <= piv]
    R = [x for x in arr[:-1] if x > piv]

    L, R = quick_sort(L), quick_sort(L)

    return L + [piv] + R


def counting_sort(arr: list) -> list:
    """This Works only for Positive int's(+ve), but can be modified for Negative's also

    TC : O(n)
    SC : O(n)"""
    len(arr)
    maxx = max(arr)
    counts = [0] * (maxx + 1)
    for x in arr:
        counts[x] += 1

    i = 0
    for c in range(maxx + 1):
        while counts[c] > 0:
            arr[i] = c
            i += 1
            counts[c] -= 1
    return arr


def main():
    inp = [1, 2, 7, -8, 34, 2, 80, 790, 6]
    counting_sort(inp)
    print("U are amazing, Keep up")


if __name__ == "__main__":
    main()
