from turtle import mode


class Node :
    def __init__(self , data , next = None):
        self.data = data
        self.next = next

class CircularLinkedList :
    def __init__(self):
        self.head = self.tail = None
        self.length = 0
        
    def insert_at_beginning(self , data):
        node = Node(data , self.head)
        if self.head is None:
            self.head = self.tail = node
            node.next = node
            self.length += 1
            return
        self.head = node
        self.tail.next = node
        self.length += 1
        
    def insert_at_end(self , data):
        node = Node(data , self.head)
        if self.head is None:
            self.head = self.tail = node
            node.next = node
            self.length += 1
            return
        self.tail.next = node
        self.tail = node
        self.length += 1
    
    def len(self):
        return self.length
    
    def pop_at_beginning(self):
        if self.head is None:
            print('List is Empty!')
            return
        self.head = self.head.next
        self.tail.next = self.head
        self.length -= 1 
      
    def pop_at_end(self):
        if self.head is None:
            print('List is Empty!')
            return
        temp = self.head
        while temp:
            if temp.next is self.tail:
                self.tail.next = None
                self.tail = temp
                temp.next = self.head
                self.length -= 1
                return 
            temp = temp.next
            
    def insert_values(self , arr : list):
        self.head = self.tail = None
        self.length = 0
        for i in arr:
            self.insert_at_end(i)
            
    def print(self):
        if self.head is None:
            print('The List is Empty!')
            return
        temp = self.head.next
        print(f'{self.head.data} ->' , end=' ')
        while temp != self.head:
            print(f'{temp.data} ->' , end=' ')
            temp = temp.next
        print(f'{self.tail.next.data}')  
    
    def insert_at(self , idx , data):
        if idx == 0:
            self.insert_at_beginning(data)
            return
        elif idx == self.length:
            self.insert_at_end(data)
            return
        elif 0 > idx or idx > self.length:
            raise Exception('Invalid Position')
            return
        pos = 0
        temp = self.head
        while temp:
            if pos == idx - 1:
                node = Node(data , temp.next)
                temp.next = node
                self.length += 1
                return
            pos += 1
            temp = temp.next 
    
    def remove_at(self , idx):
        if 0 > idx or idx >= self.length:
            raise Exception('Invalid Position')
        elif idx == 0:
            self.pop_at_beginning()
            return
        elif idx == self.length - 1:
            self.pop_at_end()
            return
        temp = self.head
        pos = 0
        while temp:
            if pos == idx - 1:
                temp.next = temp.next.next
                self.length -= 1
                return
            pos += 1
            temp = temp.next            
       
def main():
    ll = CircularLinkedList()
    ll.insert_at_end(1)        
    ll.insert_at_end(4)        
    ll.insert_at_end(3)        
    ll.insert_at_beginning(2)
    ll.insert_values([1 , 2, 3 ,4 ,5 ,6,53,3])
    # ll.pop_at_end()  
    ll.insert_at(8, 7) 
    # ll.remove_at(2)     
    ll.print()
    print(f'{ll.len() = }')



if __name__ == '__main__':
    main()