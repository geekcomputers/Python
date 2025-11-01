from tree_node import Node


def inorder_successor(root: Node) -> Node:
    """This function returns the inorder successor of a node in a BST"""

    # The inorder successor of a node is the node with the smallest value greater than the value of the node
    current: Node = root

    # The inorder successor is the leftmost node in the right subtree
    while current.left is not None:
        current = current.left
    return current
