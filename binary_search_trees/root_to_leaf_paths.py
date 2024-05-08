def print_root_to_leaf_paths(root, path):
    """ This function prints all the root to leaf paths in a BST"""
    
    # If the tree is empty, return
    if root is None:
        return
    
    # Add the root value to the path
    path.append(root.data)
    if root.left is None and root.right is None:
        print(path)
    
    # Recursively print the root to leaf paths in the left and right subtrees
    else:
        print_root_to_leaf_paths(root.left, path)
        print_root_to_leaf_paths(root.right, path)
    path.pop()