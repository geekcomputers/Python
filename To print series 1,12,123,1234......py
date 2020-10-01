def num(a):  

       # initialising starting number  

   num = 1

 # outer loop to handle number of rows  

   for i in range(0, a):  

    # re assigning num  

       num = 1

 # inner loop to handle number of columns  

           # values changing acc. to outer loop  

       for k in range(0, i+1):  

         # printing number  

           print(num, end=" ")  

            # incrementing number at each column  

           num = num + 1    

# ending line after each row  

       print("\r")  

# Driver code

a = 5

num(a)  
