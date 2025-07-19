from tree_node import Node


def is_valid_bst(root: Node | None, min_node: Node | None, max_node: Node | None) -> bool:
    """
    Function to check if a binary tree is a binary search tree.

    Args:
        root (Node | None): The root node of the binary tree. If the tree is empty, it's None.
        min_node (Node | None): The minimum value node for the current subtree.
        max_node (Node | None): The maximum value node for the current subtree.

    Returns:
        bool: True if the binary tree is a valid BST, False otherwise.
    """
    # If the tree is empty, return True
    if root is None:
        return True
    
    # If the root value is less than the minimum value or greater than the maximum value, return False
    if min_node is not None and root.data <= min_node.data or max_node is not None and root.data >= max_node.data:
        return False
    
    # Recursively check if the left and right subtrees are BSTs
    return is_valid_bst(root.left, min_node, root) and is_valid_bst(root.right, root, max_node)