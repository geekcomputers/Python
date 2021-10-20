class Node:
    """Class for node of a tree"""

    def __init__(self, info):
        """Initialising a node"""
        self.info = info
        self.left = None
        self.right = None
        # self.level = None

    def __str__(self):
        return str(self.info)

    def __del__(self):
        del self


class BinarySearchTree:
    """Class for BST"""

    def __init__(self):
        """Initialising a BST"""
        self.root = None

    def insert(self, val):
        """Creating a BST with root value as val"""
        # Check if tree has root with None value
        if self.root is None:
            self.root = Node(val)
        # Here the tree already has one root
        else:
            current = self.root
            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break

    def search(self, val, to_delete=False):
        current = self.root
        prev = -1
        while current:
            if val < current.info:
                prev = current
                current = current.left
            elif val > current.info:
                prev = current
                current = current.right
            elif current.info == val:
                if not to_delete:
                    return "Match Found"
                return prev
            else:
                break
        if not to_delete:
            return "Not Found"

    # Method to delete a tree-node if it exists, else error message will be returned.
    def delete(self, val):
        prev = self.search(val, True)
        # Check if node exists
        if prev is not None:
            # Check if node is the Root node
            if prev == -1:
                temp = self.root.left
                prev2 = None
                while temp.right:
                    prev2 = temp
                    temp = temp.right
                if prev2 is None:
                    self.root.left = temp.left
                    self.root.info = temp.info
                else:
                    prev2.right = None
                    self.root.info = temp.info
                print("Deleted Root ", val)
            # Check if node is to left of its parent
            elif prev.left and prev.left.info == val:
                # Check if node is leaf node
                if prev.left.left is prev.left.right:
                    prev.left = None
                    print("Deleted Node ", val)
                # Check if node has child at left and None at right
                elif prev.left.left and prev.left.right is None:
                    prev.left = prev.left.left
                    print("Deleted Node ", val)
                # Check if node has child at right and None at left
                elif prev.left.left is None and prev.left.right:
                    prev.left = prev.left.right
                    print("Deleted Node ", val)
                # Here node to be deleted has 2 children
                elif prev.left.left and prev.left.right:
                    temp = prev.left
                    while temp.right is not None:
                        prev2 = temp
                        temp = temp.right
                    prev2.right = None
                    prev.left.info = temp.info
                    print("Deleted Node ", val)
                else:
                    print("Error Left")

            # Check if node is to right of its parent
            elif prev.right.info == val:
                flag = 0
                # Check is node is a leaf node
                if prev.right.left is prev.right.right:
                    prev.right = None
                    flag = 1
                    print("Deleted Node ", val)
                # Check if node has left child at None at right
                if prev.right and prev.right.left and prev.right.right is None:
                    prev.right = prev.right.left
                    print("Deleted Node ", val)
                # Check if node has right child at None at left
                elif prev.right and prev.right.left is None and prev.right.right:
                    prev.right = prev.right.right
                    print("Deleted Node ", val)
                elif prev.right and prev.right.left and prev.right.right:
                    temp = prev.right
                    while temp.left is not None:
                        prev2 = temp
                        temp = temp.left
                    prev2.left = None
                    prev.right.info = temp.info
                    print("Deleted Node ", val)
                else:
                    if flag == 0:
                        print("Error")
        else:
            print("Node doesn't exists")

    def __str__(self):
        return "Not able to print tree yet"


def is_bst(node, lower_lim=None, upper_lim=None):
    """Function to find is a binary tree is a binary search tree."""
    if lower_lim is not None and node.info < lower_lim:
        return False
    if upper_lim is not None and node.info > upper_lim:
        return False
    is_left_bst = True
    is_right_bst = True
    if node.left is not None:
        is_left_bst = is_bst(node.left, lower_lim, node.info)
    if is_left_bst and node.right is not None:
        is_right_bst = is_bst(node.right, node.info, upper_lim)
    return is_left_bst and is_right_bst


def postorder(node):
    # L R N : Left , Right, Node
    if node is None:
        return
    if node.left:
        postorder(node.left)
    if node.right:
        postorder(node.right)
    print(node.info)


def inorder(node):
    # L N R : Left, Node , Right
    if node is None:
        return
    if node.left:
        inorder(node.left)
    print(node.info)
    if node.right:
        inorder(node.right)


def preorder(node):
    # N L R : Node , Left, Right
    if node is None:
        return
    print(node.info)
    if node.left:
        preorder(node.left)
    if node.right:
        preorder(node.right)


# Levelwise
def bfs(node):
    queue = []
    if node:
        queue.append(node)
    while queue != []:
        temp = queue.pop(0)
        print(temp.info)
        if temp.left:
            queue.append(temp.left)
        if temp.right:
            queue.append(temp.right)


def preorder_itr(node):
    # N L R : Node, Left , Right
    stack = [node]
    values = []
    while stack != []:
        temp = stack.pop()
        print(temp.info)
        values.append(temp.info)
        if temp.right:
            stack.append(temp.right)
        if temp.left:
            stack.append(temp.left)
    return values


def inorder_itr(node):
    # L N R : Left, Node, Right
    # 1) Create an empty stack S.
    # 2) Initialize current node as root
    # 3) Push the current node to S and set current = current->left until current is NULL
    # 4) If current is NULL and stack is not empty then
    #     a) Pop the top item from stack.
    #     b) Print the popped item, set current = popped_item->right
    #     c) Go to step 3.
    # 5) If current is NULL and stack is empty then we are done.
    stack = []
    current = node
    while True:
        if current != None:
            stack.append(current)  # L
            current = current.left
        elif stack != []:
            temp = stack.pop()
            print(temp.info)  # N
            current = temp.right  # R
        else:
            break


def postorder_itr(node):
    # L R N
    # 1. Push root to first stack.
    # 2. Loop while first stack is not empty
    # 2.1 Pop a node from first stack and push it to second stack
    # 2.2 Push left and right children of the popped node to first stack
    # 3. Print contents of second stack
    s1, s2 = [node], []
    while s1 != []:
        temp = s1.pop()
        s2.append(temp)
        if temp.left:
            s1.append(temp.left)
        if temp.right:
            s1.append(temp.right)
    print(*(s2[::-1]))


def bst_frm_pre(pre_list):
    box = Node(pre_list[0])
    if len(pre_list) > 1:
        if len(pre_list) == 2:
            if pre_list[1] > pre_list[0]:
                box.right = Node(pre_list[1])
            else:
                box.left = Node(pre_list[1])
        else:
            all_less = False
            for i in range(1, len(pre_list)):
                if pre_list[i] > pre_list[0]:
                    break
            else:
                all_less = True
            if i != 1:
                box.left = bst_frm_pre(pre_list[1:i])
            if not all_less:
                box.right = bst_frm_pre(pre_list[i:])
    return box


# Function to find the lowest common ancestor of nodes with values c1 and c2.
# It return value in the lowest common ancestor, -1 indicates value returned for None.
# Note that both values v1 and v2 should be present in the bst.
def lca(t_node, c1, c2):
    if c1 == c2:
        return c1
    current = t_node
    while current:
        if c1 < current.info and c2 < current.info:
            current = current.left
        elif c1 > current.info and c2 > current.info:
            current = current.right
        else:
            return current.info
    return -1


# Function to print element vertically which lie just below the root node
def vertical_middle_level(t_node):
    e = (t_node, 0)  # 0 indicates level 0, to left we have -ve and to right +ve
    queue = [e]
    ans = []
    # Do a level-order traversal and assign level-value to each node
    while queue != []:
        temp, level = queue.pop(0)
        if level == 0:
            ans.append(str(temp.info))
        if temp.left:
            queue.append((temp.left, level - 1))
        if temp.right:
            queue.append((temp.right, level + 1))
    return " ".join(ans)


def get_level(n, val):
    c_level = 0

    while n.info != val:
        if val < n.info:
            n = n.left
        elif val > n.info:
            n = n.right
        c_level += 1
        if n is None:
            return -1

    return c_level


def depth(node):
    if node is None:
        return 0
    l_depth, r_depth = 0, 0
    if node.left:
        l_depth = depth(node.left)
    if node.right:
        r_depth = depth(node.right)
    # print(node.info, l_depth, r_depth)
    return 1 + max(l_depth, r_depth)


t = BinarySearchTree()
t.insert(10)
t.insert(5)
t.insert(15)
t.insert(3)
t.insert(1)
t.insert(0)
t.insert(2)
t.insert(7)
t.insert(12)
t.insert(18)
t.insert(19)
print(depth(t.root))
# inorder(t.root)
# print()
# print(t.search(5))
# t.delete(7)
# t.delete(5)
# t.delete(3)
# t.delete(15)
# inorder(t.root)
# print()
# t.delete(2)
# t.delete(3)
# t.delete(7)
# t.delete(19)
# t.delete(1)
# inorder(t.root)
# b = BinarySearchTree()
# b.root = bst_frm_pre(preorder_itr(t.root))
# print(preorder_itr(b.root) == preorder_itr(t.root))
# print(lca(t.root, 3, 18))
# print(vertical_middle_level(t.root))
# print(get_level(t.root, 1))
