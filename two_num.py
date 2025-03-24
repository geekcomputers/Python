"""
Author: Anurag Kumar (mailto:anuragkumarak95@gmail.com)

This script defines a function that finds two indices in an array 
such that their corresponding values add up to a given target.

Example:
    >>> two_sum([2, 7, 11, 15], 9)
    [0, 1]

Args:
    nums (list): List of integers.
    target (int): Target sum.

Returns:
    list: Indices of the two numbers that add up to `target`.
    False: If no such pair is found.
"""

def two_sum(nums, target):
    """Finds two numbers that add up to a given target."""
    chk_map = {}
    for index, val in enumerate(nums):
        complement = target - val
        if complement in chk_map:
            return [chk_map[complement], index]
        chk_map[val] = index
    return False  # Clearer than returning `None`

# Example usage
if __name__ == "__main__":
    numbers = [2, 7, 11, 15]
    target_value = 9
    result = two_sum(numbers, target_value)
    print(result)  # Expected output: [0, 1]
