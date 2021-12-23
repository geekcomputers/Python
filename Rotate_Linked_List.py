class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Linked_List:
    def __init__(self):
        self.head = None

    def Insert_At_Beginning(self, new_data):
        """
        Inserts a new node at the beginning of the list.
        :param self: The object instance reference.
        :param new_data: The data to be inserted into the list.
        """
        new_node = Node(new_data)
        if self.head is None:
            self.head = new_node
            return
        new_node.next = self.head
        self.head = new_node

    def Rotation(self, key):
        """
        Rotate the linked list counter-clockwise by k nodes, where k is a given integer.

        :param self: The head of the linked list to be rotated.
        :type self:
        Node object or NoneType
        :param key: The number of rotations to be made. If key is greater than length of LL, then make a single rotation around the
        entire LL. 
                    If key is 0, do nothing and return None as per problem statement. 
                    Else if less than zero or not an int type
        return TypeError exception with message "Key must be an integer". 

                     Note that you have to rotate the list counter-clockwise and can't
        reverse it in place (why?).

                     You may use only constant extra space and O(1) extra time except for printing intermediate steps!
        For example if there are 7 nodes in Linked List then after 4 rotations we get following structure (7 -> 6 -> 5 -> 4 -> 3).   # noQA E501 line too long
        >80 characters; pylint disable=C0301    # noQA E501 line too long >80 characters; pylint disable=C0301    # noQA E501 line
        """
        if key == 0:
            return
        current = self.head
        count = 1
        while count < key and current is not None:
            current = current.next
            count += 1
        if current is None:
            return
        Kth_Node = current
        while current.next is not None:
            current = current.next
        current.next = self.head
        self.head = Kth_Node.next
        Kth_Node.next = None

    def Display(self):
        temp = self.head
        while(temp):
            print(temp.data, "->", end=" ")
            temp = temp.next
        print("None")


if __name__ == "__main__":
    L_list = Linked_List()
    L_list.Insert_At_Beginning(8)
    L_list.Insert_At_Beginning(5)
    L_list.Insert_At_Beginning(10)
    L_list.Insert_At_Beginning(7)
    L_list.Insert_At_Beginning(6)
    L_list.Insert_At_Beginning(11)
    L_list.Insert_At_Beginning(9)
    print("Linked List Before Rotation: ")
    L_list.Display()
    print("Linked List After Rotation: ")
    L_list.Rotation(4)
    L_list.Display()