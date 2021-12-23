class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Linked_List:
    def __init__(self):
        self.head = None

    def Insert_At_End(self, new_data):
        """
        Inserts a new node at the end of the list.
        :param self: The object pointer.
        :param new_data: The data to be inserted in the node.
        """
        new_node = Node(new_data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while(current.next):
            current = current.next
        current.next = new_node

    def Detect_and_Remove_Loop(self):
        """
        Detect and Remove Loop
        Given a linked list, check if the linked list has loop or not. Below diagram shows a linked list with a loop.
        LinkedList:
        1->2->3->4->5->6-\ 
                                      |---->6<--------| 
        If loop is present then remove the loop and return true else return false.
        NOTE : Try to solve without using extra space (Hint: Use System's Math library)
        """
        slow = fast = self.head
        while(slow and fast and fast.next):
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                self.Remove_loop(slow)
                print("Loop Found")
                return 1
        return 0

    def Remove_loop(self, Loop_node):
        """
        Removes the loop from the linked list if present.
        Parameters: Loop_node - Node at which the loop is detected.
        Returns: None
        """
        ptr1 = self.head
        while(1):
            ptr2 = Loop_node
            while(ptr2.next != Loop_node and ptr2.next != ptr1):
                ptr2 = ptr2.next
            if ptr2.next == ptr1:
                break
            ptr1 = ptr1.next
        ptr2.next = None

    def Display(self):
        temp = self.head
        while(temp):
            print(temp.data, "->", end=" ")
            temp = temp.next
        print("None")


if __name__ == "__main__":
    L_list = Linked_List()
    L_list.Insert_At_End(8)
    L_list.Insert_At_End(5)
    L_list.Insert_At_End(10)
    L_list.Insert_At_End(7)
    L_list.Insert_At_End(6)
    L_list.Insert_At_End(11)
    L_list.Insert_At_End(9)
    print("Linked List with Loop: ")
    L_list.Display()
    print("Linked List without Loop: ")
    L_list.head.next.next.next.next.next.next.next = L_list.head.next.next
    L_list.Detect_and_Remove_Loop()
    L_list.Display()
