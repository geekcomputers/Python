class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList():
    def __init__(self):
        self.head = None

    def length(self):
        curr = self.head
        count = 0
        while curr.next != None:
            count += 1
            curr = curr.next
        return count

    def add_node(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            curr = self.head
            while curr.next != None:
                curr = curr.next
            curr.next = new_node

    def insert_at_head(self, data):
        new_node = Node(data)
        temp = self.head
        self.head = new_node
        new_node.next = temp
        del temp

    def insert(self, pos, data):
        if pos < 0 or pos > self.length():
            print("Enter valid index")
        elif pos == 0:
            self.insert_at_head(data)
            return
        elif pos == self.length()-1:
            self.add_node(data)
            return
        new_node = Node(data)
        curr_pos = 0
        prev = None
        curr = self.head
        while True:
            if pos == curr_pos:
                prev.next = new_node
                new_node.next = curr
                break
            prev = curr
            curr = curr.next
            curr_pos += 1
        
    def delete_head(self):
        temp = self.head
        self.head = temp.next
        del temp
    
    def delete_end(self):
        curr = self.head
        prev = None
        while True:
            if curr.next == None:
                prev.next = None
                del curr
                break
            prev = curr
            curr = curr.next

    def delete(self, pos):
        if pos < 0 or pos > self.length():
            print("Enter valid index")
            return
        elif pos == 0:
            self.delete_head()
            return
        elif pos == self.length()-1:
            self.delete_end()
            return
        curr = self.head
        curr_pos = 0
        prev = None
        while True:
            if curr_pos == pos:
                prev.next = curr.next
                del curr
                break
            prev = curr
            curr = curr.next
            curr_pos += 1

    def display(self):
        if self.head is None:
            print("List is empty")
        rev = []
        curr = self.head
        while curr != None:
            print(f"{curr.data} --> ", end='')
            rev.append(curr.data)
            curr = curr.next
        print()
        return rev[::-1]
