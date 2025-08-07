from typing import Optional
from tree_node import Node


def print_in_range(root: Optional[Node], k1: int, k2: int) -> None:
    """This function prints the nodes in a BST that are in the range k1 to k2 inclusive"""

    # If the tree is empty, return
    if root is None:
        return

    # If the root value is in the range, print the root value
    if k1 <= root.data <= k2:
        print_in_range(root.left, k1, k2)
        print(root.data)
        print_in_range(root.right, k1, k2)

    # If the root value is less than k1, the nodes in the range will be in the right subtree
    elif root.data < k1:
        print_in_range(
            root.right, k1, k2
        )  # Fixed: original had left, which is incorrect

    # If the root value is greater than k2, the nodes in the range will be in the left subtree
    else:
        print_in_range(
            root.left, k1, k2
        )  # Fixed: original had right, which is incorrect
