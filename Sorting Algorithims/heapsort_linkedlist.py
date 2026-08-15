"""
Heap Sort on a Singly Linked List (Educational Implementation)
==============================================================

Algorithm Steps:
1. Count the number of nodes (n).
2. Build a max‑heap starting from the last internal node (n//2 - 1) up to the root.
3. Repeatedly swap the root (maximum) with the last unsorted node, then restore the heap property on the reduced heap.

Time Complexity: O(n log n) comparisons, but accessing a node by index costs O(n),
                 making the overall complexity O(n²) for this linked‑list version.
Space Complexity: O(log n) due to recursion stack (or O(1) if implemented iteratively).
Stability: Not stable.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional


class Node:
    """Singly linked list node."""

    def __init__(self, data: int) -> None:
        self.data: int = data
        self.next: Optional[Node] = None


class LinkedList:
    """Singly linked list with heap‑sort capability."""

    def __init__(self, iterable: Optional[Iterable[int]] = None) -> None:
        """Create a linked list, optionally from an iterable."""
        self.head: Optional[Node] = None
        if iterable is not None:
            for value in reversed(list(iterable)):  # maintain original order
                self.push(value)

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

    # ---------- Heap‑sort helpers ----------
    def _get_node(self, index: int) -> Optional[Node]:
        """Return the node at the given index (0‑based); return None if out of range."""
        cur = self.head
        for _ in range(index):
            if cur is None:
                return None
            cur = cur.next
        return cur

    def _swap_data(self, i: int, j: int) -> None:
        """Swap the data of two nodes by their indices."""
        node_i = self._get_node(i)
        node_j = self._get_node(j)
        if node_i is not None and node_j is not None:
            node_i.data, node_j.data = node_j.data, node_i.data

    def _heapify(self, n: int, i: int) -> None:
        """
        Maintain the max‑heap property for the subtree rooted at index i.
        Assumes that the subtrees of i are already heaps.
        n – size of the current heap (unsorted part).
        """
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        node_i = self._get_node(i)
        node_left = self._get_node(left) if left < n else None
        node_right = self._get_node(right) if right < n else None

        if (
            node_left is not None
            and node_i is not None
            and node_left.data > node_i.data
        ):
            largest = left

        node_largest = self._get_node(largest)
        if (
            node_right is not None
            and node_largest is not None
            and node_right.data > node_largest.data
        ):
            largest = right

        if largest != i:
            self._swap_data(i, largest)
            self._heapify(n, largest)  # recursive fix

    # ---------- Public sort method ----------
    def heap_sort(self) -> None:
        """
        Sort the linked list in ascending order using heap sort.

        Examples:
        >>> lst = LinkedList.from_list([4, 10, 3, 5, 1])
        >>> lst.heap_sort()
        >>> lst.to_list()
        [1, 3, 4, 5, 10]

        >>> lst = LinkedList.from_list([])
        >>> lst.heap_sort()
        >>> lst.to_list()
        []

        >>> lst = LinkedList.from_list([7])
        >>> lst.heap_sort()
        >>> lst.to_list()
        [7]

        >>> lst = LinkedList.from_list([2, 1])
        >>> lst.heap_sort()
        >>> lst.to_list()
        [1, 2]
        """
        n = len(self)
        if n <= 1:
            return

        # Build max‑heap (starting from the last internal node)
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(n, i)

        # One by one extract the maximum and place at the end
        for i in range(n - 1, 0, -1):
            self._swap_data(0, i)  # move current max to the end
            self._heapify(i, 0)  # restore heap on the reduced list


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
    ll.heap_sort()
    print("Sorted list:")
    ll.print_list()
