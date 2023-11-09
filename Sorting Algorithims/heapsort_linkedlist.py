class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def push(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def print_list(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def heapify(self, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        current = self.head
        for _ in range(i):
            current = current.next

        if left < n and current.data < current.next.data:
            largest = left

        if right < n and current.data < current.next.data:
            largest = right

        if largest != i:
            self.swap(i, largest)
            self.heapify(n, largest)

    def swap(self, i, j):
        current_i = self.head
        current_j = self.head

        for _ in range(i):
            current_i = current_i.next

        for _ in range(j):
            current_j = current_j.next

        current_i.data, current_j.data = current_j.data, current_i.data

    def heap_sort(self):
        n = 0
        current = self.head
        while current:
            n += 1
            current = current.next

        for i in range(n // 2 - 1, -1, -1):
            self.heapify(n, i)

        for i in range(n - 1, 0, -1):
            self.swap(0, i)
            self.heapify(i, 0)

# Example usage:
linked_list = LinkedList()
linked_list.push(12)
linked_list.push(11)
linked_list.push(13)
linked_list.push(5)
linked_list.push(6)
linked_list.push(7)

print("Original Linked List:")
linked_list.print_list()

linked_list.heap_sort()

print("Sorted Linked List:")
linked_list.print_list()
