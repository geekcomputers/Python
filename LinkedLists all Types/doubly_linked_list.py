'''Contains Most of the Doubly Linked List functions.\n
'variable_name' = doubly_linked_list.DoublyLinkedList() to use this an external module.\n
'variable_name'.insert_front('element') \t,'variable_name'.insert_back('element'),\n
'variable_name'.pop_front() are some of its functions.\n
To print all of its Functions use print('variable_name'.__dir__()).\n
Note:- 'variable_name' = doubly_linked_list.DoublyLinkedList() This line is Important before using any of the function.

Author :- Mugen https://github.com/Mugendesu 
'''
class Node:
    def __init__(self, val=None , next = None , prev = None):
        self.data = val
        self.next = next
        self.prev = prev

class DoublyLinkedList:
    
    def __init__(self):
        self.head = self.tail = None
        self.length = 0 

    def insert_front(self , data):
        node = Node(data , self.head)
        if self.head == None:
            self.tail = node
        node.prev = self.head
        self.head = node
        self.length += 1
        
    def insert_back(self , data):
        node = Node(data ,None, self.tail)
        if self.head == None:
            self.tail = self.head = node
            self.length += 1
        else:
            self.tail.next = node
            self.tail = node
            self.length += 1
    
    def insert_values(self , data_values : list):
        self.head = self.tail = None
        self.length = 0
        for data in data_values:
            self.insert_back(data)
    
    def pop_front(self):
        if not self.head:
            print('List is Empty!')
            return
        
        self.head = self.head.next
        self.head.prev = None
        self.length -= 1
    
    def pop_back(self):
        if not self.head:
            print('List is Empty!')
            return
        
        temp = self.tail
        self.tail = temp.prev
        temp.prev = self.tail.next = None
        self.length -= 1
    
    def print(self): 
        if self.head is None:
            print('Linked List is Empty!')
            return

        temp = self.head
        print('NULL <-' , end=' ')
        while temp:
            if temp.next == None:
                print(f'{temp.data} ->' , end = ' ')
                break
            print(f'{temp.data} <=>' , end = ' ')
            temp = temp.next
        print('NULL')
 
    def len(self):
        return self.length # O(1) length calculation
        # if self.head is None:
        #     return 0
        # count = 0
        # temp = self.head
        # while temp:
        #     count += 1
        #     temp = temp.next
        # return count
    
    def remove_at(self , idx):
        if idx < 0 or self.len() <= idx:
            raise Exception('Invalid Position')
        if idx == 0:
            self.pop_front()
            return
        elif idx == self.length -1:
            self.pop_back()
            return    
        temp = self.head
        dist = 0
        while dist != idx-1:
            dist += 1
            temp = temp.next
        temp.next = temp.next.next
        temp.next.prev = temp.next.prev.prev
        self.length -= 1
        
    def insert_at(self , idx : int , data ):
        if idx < 0 or self.len() < idx:
            raise Exception('Invalid Position')
        if idx == 0:
            self.insert_front(data)
            return
        elif idx == self.length:
            self.insert_back(data)
            return
        temp = self.head
        dist = 0
        while dist != idx-1:
            dist += 1
            temp = temp.next
        node = Node(data , temp.next , temp)
        temp.next = node
        self.length += 1
    
    def insert_after_value(self , idx_data , data):
        if not self.head : # For Empty List case
            print('List is Empty!')
            return
        
        if self.head.data == idx_data: # To insert after the Head Element 
            self.insert_at(1 , data)
            return
        temp = self.head
        while temp:
            if temp.data == idx_data:
                node = Node(data , temp.next , temp)
                temp.next = node
                self.length += 1
                return
            temp = temp.next
        print('The Element is not in the List!')
        
    def remove_by_value(self , idx_data):
        temp = self.head
        if temp.data == idx_data:
            self.pop_front()
            return
        elif self.tail.data == idx_data:
            self.pop_back()
            return
        while temp:
            if temp.data == idx_data:
                temp.prev.next = temp.next
                temp.next.prev = temp.prev
                self.length -= 1
                return
            if temp != None:
                temp = temp.next
        print("The Element is not the List!")

    def index(self , data):
        '''Returns the index of the Element'''
        if not self.head :
            print('List is Empty!')
            return
        idx = 0
        temp = self.head
        while temp:
            if temp.data == data: return idx
            temp = temp.next
            idx += 1
        print('The Element is not in the List!')

    def search(self , idx):
        '''Returns the Element at the Given Index'''
        if self.len() == 0 or idx >= self.len():
            raise Exception('Invalid Position')
            return
        temp = self.head
        curr_idx = 0
        while temp:
            if curr_idx == idx:
                return temp.data
            temp = temp.next
            curr_idx += 1
    
    def reverse(self):
        if not self.head:
            print('The List is Empty!')
            return
        prev = c_next = None
        curr = self.head
        while curr != None:
            c_next = curr.next
            curr.next = prev
            prev = curr
            curr = c_next
        self.tail = self.head
        self.head = prev
        
    def mid_element(self):
        if not self.head:
            print('List is Empty!')
            return
        slow = self.head.next
        fast = self.head.next.next
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        return slow.data

    def __dir__(self):
        funcs = ['insert_front', 'insert_back','pop_front','pop_back','print','len','length','remove_at','insert_after_value','index','search','reverse','mid_element','__dir__']
        return funcs

def main():
    ll : Node = DoublyLinkedList()
    
    ll.insert_front(1)        
    ll.insert_front(2)        
    ll.insert_front(3)
    ll.insert_back(0)
    ll.insert_values(['ZeroTwo' , 'Asuna' , 'Tsukasa' , 'Seras'])
    # ll.remove_at(3)
    # ll.insert_at(4 , 'Raeliana')
    # ll.pop_back()
    ll.insert_after_value('Asuna' , 'MaoMao')
    # print(ll.search(4))
    # ll.remove_by_value('Asuna')
    # ll.reverse()
    # print(ll.index('ZeroTwo'))
    
    ll.print()
    # print(ll.mid_element())
    # print(ll.length)
    # print(ll.__dir__())  
    
    
    
    
          
if __name__ == '__main__':
    main()