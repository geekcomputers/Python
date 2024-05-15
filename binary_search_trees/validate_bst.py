def is_valid_bst(root,min,max):
    """ Function to check if a binary tree is a binary search tree"""
    
    # If the tree is empty, return True
    if root is None:
        return True
    
    # If the root value is less than the minimum value or greater than the maximum value, return False
    if min is not None and root.data <= min.data:
        return False
    
    # If the root value is greater than the maximum value or less than the minimum value, return False
    elif max is not None and root.data >= max.data:
        return False
    
    # Recursively check if the left and right subtrees are BSTs
    return is_valid_bst(root.left,min,root) and is_valid_bst(root.right,root,max)