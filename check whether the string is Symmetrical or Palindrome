def palindrome(a): 
   
    mid = (len(a)-1)//2
    start = 0
    last = len(a)-1
    flag = 0
  
    while(start<mid): 
   
        if (a[start]== a[last]): 
              
            start += 1
            last -= 1
              
        else: 
            flag = 1
            break; 
              
    if flag == 0: 
        print("The entered string is palindrome") 
    else: 
        print("The entered string is not palindrome") 
                  
def symmetry(a): 
      
    n = len(a) 
    flag = 0
       
    if n%2: 
        mid = n//2 +1
    else: 
        mid = n//2
          
    start1 = 0
    start2 = mid 
      
    while(start1 < mid and start2 < n): 
          
        if (a[start1]== a[start2]): 
            start1 = start1 + 1
            start2 = start2 + 1
        else: 
            flag = 1
            break
      
    if flag == 0: 
        print("The entered string is symmetrical") 
    else: 
        print("The entered string is not symmetrical") 
          
string = 'amaama'
palindrome(string) 
symmetry(string) 
