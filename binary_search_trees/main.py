from tree_node import Node
from insert_in_bst import insert
from delete_a_node_in_bst import delete_node
from search_in_bst import search
from inorder_successor import inorder_successor
from mirror_a_bst import create_mirror_bst
from print_in_range import print_in_range
from root_to_leaf_paths import print_root_to_leaf_paths
from validate_bst import is_valid_bst


def main():
    
    # Create a BST
    root = None
    root = insert(root, 50)
    root = insert(root, 30)
    root = insert(root, 20)
    root = insert(root, 40)
    root = insert(root, 70)
    root = insert(root, 60)
    root = insert(root, 80)
    
    # Print the inorder traversal of the BST
    print("Inorder traversal of the original BST:")
    print_in_range(root, 10, 90)
    
    # Print the root to leaf paths
    print("Root to leaf paths:")
    print_root_to_leaf_paths(root, [])
    
    # Check if the tree is a BST
    print("Is the tree a BST:", is_valid_bst(root,None,None))
    
    
    # Delete nodes from the BST
    print("Deleting 20 from the BST:")
    root = delete_node(root, 20)
    
    # Print the inorder traversal of the BST
    print("Inorder traversal of the BST after deleting 20:")
    print_in_range(root, 10, 90)
    
    # Check if the tree is a BST
    print("Is the tree a BST:", is_valid_bst(root,None,None))
    
    
    # Delete nodes from the BST
    print("Deleting 30 from the BST:")
    root = delete_node(root, 30)
    
    # Print the inorder traversal of the BST after deleting 30
    print("Inorder traversal of the BST after deleting 30:")
    print_in_range(root, 10, 90)
    
    # Check if the tree is a BST
    print("Is the tree a BST:", is_valid_bst(root,None,None))
    
    # Delete nodes from the BST
    print("Deleting 50 from the BST:")
    root = delete_node(root, 50)
    
    # Print the inorder traversal of the BST after deleting 50
    print("Inorder traversal of the BST after deleting 50:")
    print_in_range(root, 10, 90)
    
    # Check if the tree is a BST
    print("Is the tree a BST:", is_valid_bst(root,None,None))
    
    
    print("Searching for 70 in the BST:", search(root, 70))
    print("Searching for 100 in the BST:", search(root, 100))
    print("Inorder traversal of the BST:")
    print_in_range(root, 10, 90)
    print("Creating a mirror of the BST:")
    mirror_root = create_mirror_bst(root)
    print("Inorder traversal of the mirror BST:")
    print_in_range(mirror_root, 10, 90)

if __name__ == "__main__":
    main()




