from typing import Optional
from tree_node import Node


def search(root: Optional[Node], val: int) -> bool:
    """This function searches for a node with value val in the BST and returns True if found, False otherwise"""

    # If the tree is empty, return False
    if root is None:
        return False

    # If the root value is equal to the value to be searched, return True
    if root.data == val:
        return True

    # If the value to be searched is less than the root value, search in the left subtree
    if root.data > val:
        return search(root.left, val)
    return search(root.right, val)
