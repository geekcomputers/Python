

class Node:
    def __init__(self, data: int):
        """
        Initialize a binary tree node.

        Args:
            data (int): The data stored in the node.
        """
        self.data: int = data
        self.left: Node | None = None
        self.right: Node | None = None

def inorder(root: Node | None) -> None:
    """
    This function performs an inorder traversal of a Binary Search Tree (BST).

    An inorder traversal of a BST visits the nodes in ascending order of their values.
    For each node, it first traverses the left subtree, then visits the node itself,
    and finally traverses the right subtree.

    Args:
        root (Optional[Node]): The root node of the BST. If the tree is empty, this will be None.

    Returns:
        None: This function doesn't return a value. It directly prints the node values during traversal.
    """
    # The inorder traversal of a BST visits nodes in increasing order
    if root is None:
        return
    
    # Traverse the left subtree
    inorder(root.left)
    
    # Print the root node
    print(root.data)
    
    # Traverse the right subtree
    inorder(root.right)