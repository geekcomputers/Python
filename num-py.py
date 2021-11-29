import numpy as np

# to check if shape are equal and find there power
def get_array(x, y):
    a = np.shape(x)
    b = np.shape(y)

    if a == b:
        np_pow_array = x ** y
        print("Array of powers without using np.power: ", np_pow_array)

        print("Array of powers using np.power: ", np.power(x, y))
    else:
        print("Error : Shape of the given arrays is not equal.")


# 0d array
np_arr1 = np.array(3)
np_arr2 = np.array(4)
# 1d array
np_arr3 = np.array([1, 2])
np_arr4 = np.array([3, 4])
# 2d array
np_arr5 = np.array([[1, 2], [3, 4]])
np_arr6 = np.array([[5, 6], [7, 8]])

get_array(np_arr1, np_arr2)
print()
get_array(np_arr3, np_arr4)
print()
get_array(np_arr5, np_arr6)
