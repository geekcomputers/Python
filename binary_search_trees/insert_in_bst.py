from tree_node import Node
def insert(root,val):
    
    """ This function inserts a node with value val into the BST"""
    
    # If the tree is empty, create a new node
    if root is None:
        return Node(val)
    
    # If the value to be inserted is less than the root value, insert in the left subtree
    if val < root.data:
        root.left = insert(root.left,val)
    
    # If the value to be inserted is greater than the root value, insert in the right subtree
    else:
        root.right = insert(root.right,val)
    return root