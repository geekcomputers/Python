"""
Quick Sort on a Singly Linked List (In‑Place, Recursive)
=========================================================

Algorithm Steps:
1. If the segment has 0 or 1 node, it is sorted.
2. Choose the first node's data as the pivot.
3. Partition the segment so that all elements smaller than the pivot
   come before the pivot, and all greater come after.
4. Recursively sort the left part (before pivot) and the right part (after pivot).

Time Complexity: O(n log n) average, O(n²) worst‑case (rare with random data).
Space Complexity: O(log n) recursion stack (no extra data structures).
Stability: Not stable (partition swaps elements).
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional


class Node:
    """Singly linked list node."""

    def __init__(self, data: int) -> None:
        self.data: int = data
        self.next: Optional[Node] = None


class LinkedList:
    """Singly linked list with quick‑sort capability."""

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

    # ---------- Quick‑sort helpers ----------
    @staticmethod
    def _partition(start: Optional[Node], end: Optional[Node]) -> Optional[Node]:
        """
        Partition the list segment from `start` (inclusive) to `end` (exclusive)
        using `start.data` as pivot. Returns the node that contains the pivot
        after partitioning (its final position).
        """
        if start is None or start.next is None:
            return start

        pivot = start.data
        # `prev` marks the last position where an element smaller than pivot was placed
        prev = start
        curr = start.next

        while curr is not end:
            if curr.data < pivot:
                # move prev forward and swap data
                prev = prev.next
                prev.data, curr.data = curr.data, prev.data
            curr = curr.next

        # place pivot in its correct position
        start.data, prev.data = prev.data, start.data
        return prev

    def _quick_sort(self, start: Optional[Node], end: Optional[Node]) -> None:
        """
        Recursively sort the segment from `start` to `end` (exclusive).
        """
        if start is not None and start is not end:
            pivot = LinkedList._partition(start, end)
            self._quick_sort(start, pivot)  # left part (before pivot)
            self._quick_sort(pivot.next, end)  # right part (after pivot)

    def sort(self) -> None:
        """
        Sort the linked list in ascending order using quick sort.

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
        self._quick_sort(self.head, None)


# Standalone functions for backward compatibility
def partition(start: Optional[Node], end: Optional[Node]) -> Optional[Node]:
    """Standalone partition function."""
    return LinkedList._partition(start, end)


def quicksort_LL(start: Optional[Node], end: Optional[Node]) -> None:
    """Standalone recursive quick‑sort function."""
    LinkedList()._quick_sort(start, end)


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
