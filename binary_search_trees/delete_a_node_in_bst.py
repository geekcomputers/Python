from inorder_successor import inorder_successor
from tree_node import Node


def delete_node(root: Node | None, val: int) -> Node | None:
    """
    This function deletes a node with value val from the BST.

    Args:
        root (Node | None): The root node of the binary search tree.
        If the tree is empty, it's None.
        val (int): The value of the node to be deleted.

    Returns:
        Node | None: The root node of the binary search tree after deleting the node. 
        If the tree becomes empty, returns None.
    """
    # Search in the left subtree
    if root and root.data < val:
        root.right = delete_node(root.right, val)

    # Search in the right subtree
    elif root and root.data > val:
        root.left = delete_node(root.left, val)

    # Node to be deleted is found
    elif root:
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
        IS: Node = inorder_successor(root.right)
        root.data = IS.data
        root.right = delete_node(root.right, IS.data)
    return root
