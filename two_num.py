"""
Author: Anurag Kumar (mailto:anuragkumarak95@gmail.com)

Description:
    This function finds two numbers in a given list that add up to a specified target. 
    It returns the indices of those two numbers.

Constraints:
    - Each input will have exactly one solution.
    - The same element cannot be used twice.

Example:
    >>> two_sum([2, 7, 11, 15], 9)
    [0, 1]
"""

from typing import List, Optional

def two_sum(nums: List[int], target: int) -> Optional[List[int]]:
    """
    Finds indices of two numbers in 'nums' that add up to 'target'.

    Args:
        nums (List[int]): List of integers.
        target (int): Target sum.

    Returns:
        Optional[List[int]]: Indices of the two numbers that add up to the target,
                             or None if no such pair is found.
    """
    if len(nums) < 2:
        raise ValueError("Input list must contain at least two numbers.")
    
    if not all(isinstance(num, int) for num in nums):
        raise TypeError("All elements in the list must be integers.")
    
    # Dictionary to track seen values and their indices
    seen_values = {}

    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen_values:
            return [seen_values[complement], index]
        seen_values[value] = index

    return None

# Example usage
if __name__ == "__main__":
    example_nums = [2, 7, 11, 15]
    example_target = 9
    result = two_sum(example_nums, example_target)

    if result:
        num1, num2 = example_nums[result[0]], example_nums[result[1]]
        print(f"Indices that add up to {example_target}: {result} (Values: {num1} + {num2})")
    else:
        print(f"No combination found that adds up to {example_target}.")
