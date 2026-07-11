"""
Merge Sort on a Singly Linked List (Standard Implementation)
=============================================================

Algorithm Steps:
1. If the list is empty or has one node, it is already sorted.
2. Find the middle node using the slow/fast pointer technique.
3. Split the list into two halves.
4. Recursively sort each half.
5. Merge the two sorted halves back together.

Time Complexity: O(n log n)
Space Complexity: O(log n) for recursion stack (in‑place node rearrangement).
Stability: Stable (equal elements retain their relative order).
"""

from __future__ import annotations
from typing import Optional, Iterable, List, Iterator


class Node:
    """Singly linked list node."""

    def __init__(self, data: int) -> None:
        self.data: int = data
        self.next: Optional[Node] = None


class LinkedList:
    """Singly linked list with merge‑sort capability."""

    def __init__(self, iterable: Optional[Iterable[int]] = None) -> None:
        """Create a linked list, optionally from an iterable."""
        self.head: Optional[Node] = None
        if iterable is not None:
            for value in iterable:
                self.append(value)

    # ---------- Basic list operations ----------
    def push(self, data: int) -> None:
        """Insert a new node at the head."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def append(self, data: int) -> None:
        """Insert a new node at the tail."""
        if not self.head:
            self.head = Node(data)
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = Node(data)

    def to_list(self) -> List[int]:
        """Convert the linked list to a Python list."""
        result: List[int] = []
        cur = self.head
        while cur:
            result.append(cur.data)
            cur = cur.next
        return result

    @classmethod
    def from_list(cls, lst: List[int]) -> LinkedList:
        """Build a linked list from a Python list (preserving order)."""
        ll = cls()
        for val in lst:
            ll.append(val)
        return ll

    def __len__(self) -> int:
        """Return the number of nodes."""
        count = 0
        cur = self.head
        while cur:
            count += 1
            cur = cur.next
        return count

    def __iter__(self) -> Iterator[int]:
        """Iterate over node data."""
        cur = self.head
        while cur:
            yield cur.data
            cur = cur.next

    def print_list(self) -> None:
        """Print the list in a human‑readable format."""
        cur = self.head
        while cur:
            print(cur.data, end=" -> ")
            cur = cur.next
        print("None")

    # ---------- Merge‑sort implementation ----------
    @staticmethod
    def _merge(left: Optional[Node], right: Optional[Node]) -> Optional[Node]:
        """
        Merge two sorted linked lists and return the head of the merged list.
        This is a static helper that works on raw Node objects.
        """
        if left is None:
            return right
        if right is None:
            return left

        if left.data <= right.data:  # stable sort: use <= to preserve order
            result = left
            result.next = LinkedList._merge(left.next, right)
        else:
            result = right
            result.next = LinkedList._merge(left, right.next)
        return result

    def _merge_sort(self, head: Optional[Node]) -> Optional[Node]:
        """
        Recursively sort the list starting at 'head' and return the new head.
        This internal method works directly on nodes.
        """
        if head is None or head.next is None:
            return head

        # Find the middle node using slow/fast pointers
        slow = head
        fast = head.next
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next

        left = head
        right = slow.next
        slow.next = None  # split the list

        # Recursively sort both halves
        left = self._merge_sort(left)
        right = self._merge_sort(right)

        # Merge the sorted halves
        return LinkedList._merge(left, right)

    def sort(self) -> None:
        """
        Sort the linked list in ascending order using merge sort.

        Examples:
        >>> lst = LinkedList.from_list([4, 10, 3, 5, 1])
        >>> lst.sort()
        >>> lst.to_list()
        [1, 3, 4, 5, 10]

        >>> lst = LinkedList.from_list([])
        >>> lst.sort()
        >>> lst.to_list()
        []

        >>> lst = LinkedList.from_list([7])
        >>> lst.sort()
        >>> lst.to_list()
        [7]

        >>> lst = LinkedList.from_list([2, 1])
        >>> lst.sort()
        >>> lst.to_list()
        [1, 2]
        """
        self.head = self._merge_sort(self.head)


# Alias for backward compatibility with the original file
def merge_sort(head: Optional[Node]) -> Optional[Node]:
    """Standalone function that sorts a list starting at 'head'."""
    return LinkedList()._merge_sort(head)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Interactive example
    ll = LinkedList()
    print("Enter space‑separated integers to insert at the head (push):")
    data = list(map(int, input().split()))
    for val in data:
        ll.push(val)  # push reverses the order
    print("Original list (head first):")
    ll.print_list()
    ll.sort()
    print("Sorted list:")
    ll.print_list()
