from typing import Optional
from inorder_successor import inorder_successor
from tree_node import Node


# The above line imports the inorder_successor function from the inorder_successor.py file
def delete_node(root: Node, val: int) -> Optional[Node]:
    """This function deletes a node with value val from the BST"""

    # Search in the right subtree
    if root.data < val:
        root.right = delete_node(root.right, val)

    # Search in the left subtree
    elif root.data > val:
        root.left = delete_node(root.left, val)

    # Node to be deleted is found
    else:
        # Case 1: No child (leaf node)
        if root.left is None and root.right is None:
            return None

        # Case 2: One child
        if root.left is None:
            return root.right

        # Case 2: One child
        elif root.right is None:
            return root.left

        # Case 3: Two children
        # Find the inorder successor
        is_node: Node = inorder_successor(root.right)
        root.data = is_node.data
        root.right = delete_node(root.right, is_node.data)
    return root
