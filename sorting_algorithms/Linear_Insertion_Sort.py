def Linear_Search(Test_arr, val):
    index = 0
    for i in range(len(Test_arr)):
        if val > Test_arr[i]:
            index = i + 1
    return index


def Insertion_Sort(Test_arr):
    for i in range(1, len(Test_arr)):
        val = Test_arr[i]
        j = Linear_Search(Test_arr[:i], val)
        Test_arr.pop(i)
        Test_arr.insert(j, val)
    return Test_arr


if __name__ == "__main__":
    Test_list = input("Enter the list of Numbers: ").split()
    Test_list = [int(i) for i in Test_list]
    print(f"Binary Insertion Sort: {Insertion_Sort(Test_list)}")
