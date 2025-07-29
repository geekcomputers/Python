from tree_node import Node


def print_root_to_leaf_paths(root: Node | None, path: list[int]) -> None:
    """
    This function prints all the root-to-leaf paths in a BST.

    Args:
        root (Node | None): The root node of the binary search tree.
        If the tree is empty, it's None.
        path (List[int]): A list to store the current path from the root to a leaf.

    Returns:
        None
    """
    # If the tree is empty, return
    if root is None:
        return

    # Add the root value to the path
    path.append(root.data)
    if root.left is None and root.right is None:
        print(path)

    # Recursively print the root-to-leaf paths in the left and right subtrees
    else:
        print_root_to_leaf_paths(root.left, path)
        print_root_to_leaf_paths(root.right, path)
    path.pop()
