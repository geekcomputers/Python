from typing import Optional


# Node class for binary tree
class Node:
    def __init__(self, data: int) -> None:
        self.data: int = data
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
