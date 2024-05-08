def inorder(root):
    """ This function performs an inorder traversal of a BST"""
    
    # The inorder traversal of a BST is the nodes in increasing order
    if root is None:
        return
    
    # Traverse the left subtree
    inorder(root.left)
    
    # Print the root node
    print(root.data)
    
    # Traverse the right subtree
    inorder(root.right)