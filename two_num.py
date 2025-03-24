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

from typing import List, Union

def two_sum(nums: List[int], target: int) -> Union[List[int], bool]:
    """
    Finds indices of two numbers in 'nums' that add up to 'target'.

    Args:
        nums (List[int]): List of integers.
        target (int): Target sum.

    Returns:
        List[int]: Indices of the two numbers that add up to the target.
        False: If no such pair is found.
    """
    # Dictionary to track seen values and their indices
    seen_values = {}

    for index, value in enumerate(nums):
        complement = target - value
        
        # Check if the complement exists in the dictionary
        if complement in seen_values:
            return [seen_values[complement], index]

        # Add current value to dictionary for future reference
        seen_values[value] = index

    # Return False if no pair is found (explicit is better than implicit)
    return False

# Example usage
if __name__ == "__main__":
    example_nums = [2, 7, 11, 15]
    example_target = 9
    result = two_sum(example_nums, example_target)
    
    # Clean, professional result display
    if result:
        print(f"Indices that add up to {example_target}: {result}")
    else:
        print(f"No combination found that adds up to {example_target}.")
