"""
NumPy Array Exponentiation

Check if two arrays have the same shape and compute element-wise powers
with and without np.power.

Example usage:
>>> import numpy as np
>>> x = np.array([1, 2])
>>> y = np.array([3, 4])
>>> get_array(x, y)  # doctest: +ELLIPSIS
Array of powers without using np.power:  [ 1 16]
Array of powers using np.power:  [ 1 16]
"""

import numpy as np


def get_array(x: np.ndarray, y: np.ndarray) -> None:
    """
    Compute element-wise power of two NumPy arrays if their shapes match.

    Parameters
    ----------
    x : np.ndarray
        Base array.
    y : np.ndarray
        Exponent array.

    Returns
    -------
    None
        Prints the element-wise powers using both operator ** and np.power.

    Example:
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[2, 2], [2, 2]])
    >>> get_array(a, b)  # doctest: +ELLIPSIS
    Array of powers without using np.power:  [[ 1  4]
     [ 9 16]]
    Array of powers using np.power:  [[ 1  4]
     [ 9 16]]
    """
    if x.shape == y.shape:
        np_pow_array = x**y
        print("Array of powers without using np.power: ", np_pow_array)
        print("Array of powers using np.power: ", np.power(x, y))
    else:
        print("Error: Shape of the given arrays is not equal.")


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # 0D array
    np_arr1 = np.array(3)
    np_arr2 = np.array(4)
    # 1D array
    np_arr3 = np.array([1, 2])
    np_arr4 = np.array([3, 4])
    # 2D array
    np_arr5 = np.array([[1, 2], [3, 4]])
    np_arr6 = np.array([[5, 6], [7, 8]])

    get_array(np_arr1, np_arr2)
    print()
    get_array(np_arr3, np_arr4)
    print()
    get_array(np_arr5, np_arr6)
