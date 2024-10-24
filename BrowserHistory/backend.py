class DLL:
    """
        a doubly linked list that holds the current page,
        next page, and previous page.
        Used to enforce order in operations
    """
    def __init__(self, val: str =None):
        self.val = val
        self.nxt = None
        self.prev = None


class BrowserHistory:
    """
    This class designs the operations of a browser history

    It works by using a doubly linked list to hold the urls
    """

    def __init__(self, homepage: str):
        """
        Returns - None
        Input - None
        ----------
        - Initialize doubly linked list which will serve as the
            browser history and sets the current page
        """
        self.head = DLL(homepage)
        self.curr = self.head
        
    def visit(self, url: str) -> None:
        """
        Returns - None
        Input - str
        ----------
        - Adds the current url to the DLL
        - sets both the next and previous values
        """
        url_node = DLL(url)
        self.curr.nxt = url_node
        url_node.prev = self.curr
        
        self.curr = url_node
        

    def back(self, steps: int) -> str:
        """
        Returns - str
        Input - int
        ----------
        - Iterates through the DLL backwards `step` number of times
        - returns the appropriate value
        """
        while steps > 0 and self.curr.prev:
            self.curr = self.curr.prev
            steps -= 1
        return self.curr.val
        

    def forward(self, steps: int) -> str:
        """
        Returns - str
        Input - int
        ----------
        - Iterates through the DLL forewards `step` number of times
        - returns the appropriate value
        """
        while steps > 0 and self.curr.nxt:
            self.curr = self.curr.nxt
            steps -= 1
        return self.curr.val
        

if __name__ == "__main__":
    obj = BrowserHistory("google.com")
    obj.visit("twitter.com")
    param_2 = obj.back(1)
    param_3 = obj.forward(1)

    print(param_2)
    print(param_3)