class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def insert_at_beginning(self, new_data):
        new_node = Node(new_data)
        if self.head is None:
            self.head = new_node
            return
        new_node.next = self.head
        self.head = new_node

    def add_two_no(self, first, second):
        prev = None
        temp = None
        carry = 0
        while first is not None or second is not None:
            first_data = 0 if first is None else first.data
            second_data = 0 if second is None else second.data
            Sum = carry + first_data + second_data
            carry = 1 if Sum >= 10 else 0
            Sum = Sum if Sum < 10 else Sum % 10
            temp = Node(Sum)
            if self.head is None:
                self.head = temp
            else:
                prev.next = temp
            prev = temp
            if first is not None:
                first = first.next
            if second is not None:
                second = second.next
        if carry > 0:
            temp.next = Node(carry)

    def __str__(self):
        temp = self.head
        while temp:
            print(temp.data, "->", end=" ")
            temp = temp.next
        return "None"


if __name__ == "__main__":
    first = LinkedList()
    second = LinkedList()
    first.insert_at_beginning(6)
    first.insert_at_beginning(4)
    first.insert_at_beginning(9)

    second.insert_at_beginning(2)
    second.insert_at_beginning(2)

    print("First Linked List: ")
    print(first)
    print("Second Linked List: ")
    print(second)

    result = LinkedList()
    result.add_two_no(first.head, second.head)
    print("Final Result: ")
    print(result)
