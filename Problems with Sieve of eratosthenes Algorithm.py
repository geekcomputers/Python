#!/usr/bin/env python
# coding: utf-8

# In[55]:


# prime number till n
def prime(n):
    p=[True for i in range(n+1)]
#    i=2
#     while(i*i<=n):
    for i in range(2,int(n**1/2)):
        
        if p[i]==True:
            
            for j in range(i*i,n+1,i):
                
                p[j]=False
     
#         i+=1
    for i in range(2,n+1):
        if p[i]:
            print(i) 
        
            
        
n=int(input())
prime(n)


# In[43]:


# two prime whose sum is equal to n
def prime(n,p):
#     p[0]=p[1]==False
#     for i in range(2, n+1): 
# 	     p[i] = True
    
    i=2
    while(i*i<=n):
#     for i in range(2,int(n**1/2)):
        
        if p[i]==True:
            
#             j=i*i
#             while(i<=n):
            for j in range(i*i,n+1,i):
                
                p[j]=False
#                 j=+i
     
        i+=1
def printp(n):
    p=[True for i in range(n+1)]
#     p=[0]*(n+1)
    prime(n,p)
    for i in range(0,n):
        if p[i] and p[n-i]:
            print(i,n-i)
            return
            
        
n=int(input())
printp(n)
        
        


# #

# In[49]:


# largest prime number till n
def prime(n,p):
#     p[0]=p[1]==False
#     for i in range(2, n+1): 
# 	     p[i] = True
    
    i=2
    while(i*i<=n):
#     for i in range(2,int(n**1/2)):
        
        if p[i]==True:
            
#             j=i*i
#             while(i<=n):
            for j in range(i*i,n+1,i):
                
                p[j]=False
#                 j=+i
     
        i+=1
def printp(n):
    p=[True for i in range(n+1)]
#     p=[0]*(n+1)
    prime(n,p)
    for i in range(n,2,-1):
        if p[i] :
            print(i)
            return
            
        
n=int(input())
printp(n)
        
        


# In[54]:


# two distict prime number whose multiply is equal to n
def prime(n,p):
#     p[0]=p[1]==False
#     for i in range(2, n+1): 
# 	     p[i] = True
    
    i=2
    while(i*i<=n):
#     for i in range(2,int(n**1/2)):
        
        if p[i]==True:
            
#             j=i*i
#             while(i<=n):
            for j in range(i*i,n+1,i):
                
                p[j]=False
#                 j=+i
     
        i+=1
def printp(n):
    p=[True for i in range(n+1)]
#     p=[0]*(n+1)
    c=0
    prime(n,p)
    for i in range(2,n):
        x=int(n/i)
        
        if p[i] and p[x] and x!=i and i*x==n :
            print(i,"",x)
            c=1
            return
    if c==0:
        print("No pair")
            
        
n=int(input())
printp(n)
        
        


# In[ ]:





# In[ ]:




