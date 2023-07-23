def Binary_Search(Test_arr, low, high, k):
    if high >= low:
        Mid = (low + high) // 2
        if Test_arr[Mid] < k:
            return Binary_Search(Test_arr, Mid + 1, high, k)
        elif Test_arr[Mid] > k:
            return Binary_Search(Test_arr, low, Mid - 1, k)
        else:
            return Mid
    else:
        return low


def Insertion_Sort(Test_arr):
    for i in range(1, len(Test_arr)):
        val = Test_arr[i]
        j = Binary_Search(Test_arr[:i], 0, len(Test_arr[:i]) - 1, val)
        Test_arr.pop(i)
        Test_arr.insert(j, val)
    return Test_arr


if __name__ == "__main__":
    Test_list = input("Enter the list of Numbers: ").split()
    Test_list = [int(i) for i in Test_list]
    print(f"Binary Insertion Sort: {Insertion_Sort(Test_list)}")
