#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add two numbers represented as linked lists.

Each linked list stores digits in reverse order (head is the least significant digit).
This allows straightforward addition with carry propagation.

Example:
    Number 946 is stored as 6 -> 4 -> 9
    Number 22  is stored as 2 -> 2
    Sum 968   is stored as 8 -> 6 -> 9
"""

from typing import Optional


class Node:
    """Singly linked list node."""

    def __init__(self, data: int) -> None:
        self.data: int = data
        self.next: Optional["Node"] = None


class LinkedList:
    """A singly linked list with head pointing to the least significant digit."""

    def __init__(self) -> None:
        self.head: Optional[Node] = None

    def append(self, data: int) -> None:
        """
        Append a new node with `data` at the end (tail).
        This keeps the head as the least significant digit.
        """
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        curr = self.head
        while curr.next is not None:
            curr = curr.next
        curr.next = new_node

    @classmethod
    def from_number(cls, num: int) -> "LinkedList":
        """
        Create a linked list representing the digits of `num`
        with head as the least significant digit.

        >>> lst = LinkedList.from_number(946)
        >>> lst.head.data
        6
        >>> lst.head.next.data
        4
        >>> lst.head.next.next.data
        9
        """
        lst = cls()
        if num == 0:
            lst.append(0)
            return lst
        while num > 0:
            lst.append(num % 10)
            num //= 10
        return lst

    def to_number(self) -> int:
        """
        Convert the linked list back to an integer.

        >>> LinkedList.from_number(946).to_number()
        946
        """
        result = 0
        multiplier = 1
        curr = self.head
        while curr is not None:
            result += curr.data * multiplier
            multiplier *= 10
            curr = curr.next
        return result

    def __str__(self) -> str:
        """Display the linked list from head (least significant) to tail."""
        parts = []
        curr = self.head
        while curr is not None:
            parts.append(str(curr.data))
            curr = curr.next
        return " -> ".join(parts) if parts else "Empty"

    def __repr__(self) -> str:
        return f"LinkedList({self})"


def add_two_numbers(l1: LinkedList, l2: LinkedList) -> LinkedList:
    """
    Add two numbers represented by linked lists (head = least significant digit).

    Returns a new LinkedList representing the sum.

    Examples:
        >>> l1 = LinkedList.from_number(946)
        >>> l2 = LinkedList.from_number(22)
        >>> result = add_two_numbers(l1, l2)
        >>> result.to_number()
        968
        >>> str(result)
        '8 -> 6 -> 9'
    """
    dummy = Node(0)  # Sentinel node
    tail = dummy
    carry = 0
    p = l1.head
    q = l2.head

    while p is not None or q is not None or carry:
        val1 = p.data if p else 0
        val2 = q.data if q else 0
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10

        tail.next = Node(digit)
        tail = tail.next

        if p:
            p = p.next
        if q:
            q = q.next

    result = LinkedList()
    result.head = dummy.next
    return result


if __name__ == "__main__":
    # Demonstration
    first = LinkedList.from_number(946)
    second = LinkedList.from_number(22)

    print("First Linked List (head = least significant):")
    print(first)  # 6 -> 4 -> 9

    print("Second Linked List:")
    print(second)  # 2 -> 2

    result = add_two_numbers(first, second)
    print("Sum (head = least significant):")
    print(result)  # 8 -> 6 -> 9

    print(f"Sum as integer: {result.to_number()}")  # 968

    # Run doctests
    import doctest

    doctest.testmod(verbose=True)
