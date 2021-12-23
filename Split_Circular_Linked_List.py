class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Circular_Linked_List:
    def __init__(self):
        self.head = None

    def Push(self, data):
        """
        This function takes a data as an argument and creates a new node with the given data.
        It then assigns the next value of the new node to be equal to
        head. If there is no head, it assigns None to temp1 and sets temp1's next value equal to itself.
        """
        temp = Node(data)
        temp.next = self.head
        temp1 = self.head
        if self.head is not None:
            while temp1.next is not None:
                temp1 = temp1.next
            temp1.next = temp
        else:
            temp.next = temp
        self.head = temp

    def Split_List(self, head1, head2):
        """
        This function takes in the head of a linked list and two new heads for two separate lists.
        It first checks if the linked list is empty or not, if it
        is then it returns None.
        If not, then it finds the middle node of the linked list by moving fast_ptr twice as fast as slow_ptr and then assigns that
        node to slow_ptr.
        It also makes sure that there are at least 2 nodes in this linked list before assigning head2 to be equal to slow_ptr's next value
        (the second half). 
        Finally, we assign slow ptr's next value to be equal to head1 which means that we have split our original ll into two separate lls
        with one having all odd nodes and other having all even nodes.
        """
        if self.head is None:
            return
        slow_ptr = self.head
        fast_ptr = self.head
        while fast_ptr.next != self.head and fast_ptr.next.next != self.head:
            fast_ptr = fast_ptr.next.next
            slow_ptr = slow_ptr.next.next
        if fast_ptr.next.next == self.head:
            fast_ptr = fast_ptr.next
        head1 = self.head
        slow_ptr.next = head1
        if self.head.next != self.head:
            head2.head = slow_ptr.next
        fast_ptr.next = slow_ptr.next

    def Display(self):
        """
        Display()

        Prints the linked list in forward direction.

            Args: None

            Returns: None
        """
        temp = self.head
        if self.head is not None:
            while(temp):
                print(temp.data, "->", end=" ")
                temp = temp.next
                if temp == self.head:
                    print(temp.data)
                    break


if __name__ == "__main__":

    L_list = Circular_Linked_List()
    head1 = Circular_Linked_List()
    head2 = Circular_Linked_List()
    L_list.Push(6)
    L_list.Push(4)
    L_list.Push(2)
    L_list.Push(8)
    L_list.Push(12)
    L_list.Push(10)
    L_list.Split_List(head1, head2)
    print("Circular Linked List: ")
    L_list.Display()
    print("Firts Split Linked List: ")
    head1.Display()
    print("Second Split Linked List: ")
    head2.Display()
