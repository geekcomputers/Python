from __future__ import annotations


class Node:
    def __init__(self, data: int) -> None:
        """
        Initialize a binary tree node.

        Args:
            data (int): The data stored in the node.
        """
        self.data: int = data
        self.left: Node | None = None
        self.right: Node | None = None