from typing import Optional
from tree_node import Node


def create_mirror_bst(root: Optional[Node]) -> Optional[Node]:
    """Function to create a mirror of a binary search tree"""

    # If the tree is empty, return None
    if root is None:
        return None

    # Recursively create the mirror of the left and right subtrees
    left_mirror: Optional[Node] = create_mirror_bst(root.left)
    right_mirror: Optional[Node] = create_mirror_bst(root.right)

    # Swap left and right subtrees
    root.left = right_mirror
    root.right = left_mirror
    return root
