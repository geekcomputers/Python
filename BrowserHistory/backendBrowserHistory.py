class DLL:
    """
        a doubly linked list that holds the current page,
        next page, and previous page.
        Used to enforce order in operations.
    """
    def __init__(self, val: str =None):
        self.val = val
        self.nxt = None
        self.prev = None


class BrowserHistory:
    """
    This class designs the operations of a browser history

    It works by using a doubly linked list to hold the urls with optimized
    navigation using step counters and memory management
    """

    def __init__(self, homepage: str):
        """
        Returns - None
        Input - str
        ----------
        - Initialize doubly linked list which will serve as the
            browser history and sets the current page
        - Initialize navigation counters
        """
        self._head = DLL(homepage)
        self._curr = self._head
        self._back_count = 0
        self._forward_count = 0
        
    def visit(self, url: str) -> None:
        """
        Returns - None
        Input - str
        ----------
        - Adds the current url to the DLL
        - Sets both the next and previous values
        - Cleans up forward history to prevent memory leaks
        - Resets forward count and increments back count
        """
        # Clear forward history to prevent memory leaks
        self._curr.nxt = None
        self._forward_count = 0
        
        # Create and link new node
        url_node = DLL(url)
        self._curr.nxt = url_node
        url_node.prev = self._curr
        
        # Update current node and counts
        self._curr = url_node
        self._back_count += 1

    def back(self, steps: int) -> str:
        """
        Returns - str
        Input - int
        ----------
        - Moves backwards through history up to available steps
        - Updates navigation counters
        - Returns current page URL
        """
        # Only traverse available nodes
        steps = min(steps, self._back_count)
        while steps > 0:
            self._curr = self._curr.prev
            steps -= 1
            self._back_count -= 1
            self._forward_count += 1
        return self._curr.val

    def forward(self, steps: int) -> str:
        """
        Returns - str
        Input - int
        ----------
        - Moves forward through history up to available steps
        - Updates navigation counters
        - Returns current page URL
        """
        # Only traverse available nodes
        steps = min(steps, self._forward_count)
        while steps > 0:
            self._curr = self._curr.nxt
            steps -= 1
            self._forward_count -= 1
            self._back_count += 1
        return self._curr.val
        

if __name__ == "__main__":
    obj = BrowserHistory("google.com")
    obj.visit("twitter.com")
    param_2 = obj.back(1)
    param_3 = obj.forward(1)

    print(param_2)
    print(param_3)
