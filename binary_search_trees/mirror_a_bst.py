from tree_node import Node
def create_mirror_bst(root):
    """ Function to create a mirror of a binary search tree"""
    
    # If the tree is empty, return None
    if root is None:
        return None
    
    # Create a new node with the root value
    
    # Recursively create the mirror of the left and right subtrees
    left_mirror = create_mirror_bst(root.left)
    right_mirror = create_mirror_bst(root.right)
    root.left = right_mirror
    root.right = left_mirror
    return root