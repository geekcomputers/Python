from tree_node import Node


def create_mirror_bst(root: Node | None) -> Node | None:
    """
    Function to create a mirror of a binary search tree.

    Args:
        root (Node | None): The root node of the binary search tree. If the tree is empty, it's None.

    Returns:
        Node | None: The root node of the mirrored binary search tree. If the original tree is empty, returns None.
    """
    # If the tree is empty, return None
    if root is None:
        return None

    # Recursively create the mirror of the left and right subtrees
    left_mirror: Node | None = create_mirror_bst(root.left)
    right_mirror: Node | None = create_mirror_bst(root.right)
    root.left = right_mirror
    root.right = left_mirror
    return root
