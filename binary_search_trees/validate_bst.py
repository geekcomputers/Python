from typing import Optional
from tree_node import Node


def is_valid_bst(
    root: Optional[Node], min_node: Optional[Node], max_node: Optional[Node]
) -> bool:
    """Function to check if a binary tree is a binary search tree"""

    # If the tree is empty, return True
    if root is None:
        return True

    # If the root value is less than or equal to the minimum value, return False
    if min_node is not None and root.data <= min_node.data:
        return False

    # If the root value is greater than or equal to the maximum value, return False
    if max_node is not None and root.data >= max_node.data:
        return False

    # Recursively check if the left and right subtrees are BSTs
    return is_valid_bst(root.left, min_node, root) and is_valid_bst(
        root.right, root, max_node
    )
