def print_in_range(root,k1,k2):
  
  """ This function prints the nodes in a BST that are in the range k1 to k2 inclusive"""
  
  # If the tree is empty, return
  if root is None:
    return
  
  # If the root value is in the range, print the root value
  if root.data >= k1 and root.data <= k2:
    print_in_range(root.left,k1,k2)
    print(root.data)
    print_in_range(root.right,k1,k2)
    
  # If the root value is less than k1, the nodes in the range will be in the right subtree
  elif root.data < k1:
    print_in_range(root.left,k1,k2)
    
  # If the root value is greater than k2, the nodes in the range will be in the left subtree
  else:
    print_in_range(root.right,k1,k2)