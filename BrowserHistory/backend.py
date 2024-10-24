class DLL:
    def __init__(self, val: str =None):
        self.val = val
        self.nxt = None
        self.prev = None


class BrowserHistory:
    """
        This class designs the operations of a
        broswer history
    """

    def __init__(self, homepage: str):
        self.head = DLL(homepage)
        self.curr = self.head
        
    def visit(self, url: str) -> None:
        url_node = DLL(url)
        self.curr.nxt = url_node
        url_node.prev = self.curr
        
        self.curr = url_node
        

    def back(self, steps: int) -> str:
        while steps > 0 and self.curr.prev:
            self.curr = self.curr.prev
            steps -= 1
        return self.curr.val
        

    def forward(self, steps: int) -> str:
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