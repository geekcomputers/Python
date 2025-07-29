from tree_node import Node


def insert(root: Node | None, val: int) -> Node:
    """
    This function inserts a node with value val into the BST.

    Args:
        root (Node | None): The root node of the binary search tree.
        If the tree is empty, it's None.
        val (int): The value of the node to be inserted.

    Returns:
        Node: The root node of the binary search tree after inserting the node.
    """
    # If the tree is empty, create a new node
    if root is None:
        return Node(val)

    # If the value to be inserted is less than the root value,
    # insert it into the left subtree
    if val < root.data:
        root.left = insert(root.left, val)

    # If the value to be inserted is greater than the root value,
    # insert it into the right subtree
    else:
        root.right = insert(root.right, val)
    return root
