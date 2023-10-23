from __future__ import annotations

class Node:
    def __init__(self, data: int) -> None:
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, new_data: int) -> None:
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node

    def printLL(self) -> None:
        temp = self.head
        if temp == None:
            return 'Linked List is empty'
        while temp.next:
            print(temp.data, '->', end='')
            temp = temp.next
        print(temp.data)
        return

# Merge two sorted linked lists
def merge(left, right):
    if not left:
        return right
    if not right:
        return left

    if left.data < right.data:
        result = left
        result.next = merge(left.next, right)
    else:
        result = right
        result.next = merge(left, right.next)

    return result

# Merge sort for linked list
def merge_sort(head):
    if not head or not head.next:
        return head

    # Find the middle of the list
    slow = head
    fast = head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    left = head
    right = slow.next
    slow.next = None

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)

if __name__ == "__main__":
    ll = LinkedList()
    print("Enter the space-separated values of numbers to be inserted in the linked list prompted below:")
    arr = list(map(int, input().split()))
    for num in arr:
        ll.insert(num)

    print("Linked list before sorting:")
    ll.printLL()

    ll.head = merge_sort(ll.head)

    print('Linked list after sorting:')
    ll.printLL()
