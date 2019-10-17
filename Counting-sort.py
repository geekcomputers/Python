#python program for counting sort (updated)
#import some libraries
import random, math
def get_sortkey(n):
    # Define the method to retrieve the key 
    return n

def counting_sort(tlist, k, get_sortkey):
    """ Counting sort algo with sort in place.
        Args: 
            tlist: target list to sort
            k: max value assume known before hand
            get_sortkey: function to retrieve the key that is apply to elements of tlist to be used in the count list index.
            map info to index of the count list.
        Adv:
            The count (after cum sum) will hold the actual position of the element in sorted order
            Using the above, 
            
    """

    # Create a count list and using the index to map to the integer in tlist.
    count_list = [0]*(k)

    # iterate the tgt_list to put into count list
    for n in tlist:
        count_list[get_sortkey(n)] = count_list[get_sortkey(n)] + 1  
        
    # Modify count list such that each index of count list is the combined sum of the previous counts 
    # each index indicate the actual position (or sequence) in the output sequence.
    for i in range(k):
        if i ==0:
            count_list[i] = count_list[i]
        else:
            count_list[i] += count_list[i-1]
 

    output = [None]*len(tlist)
    for i in range(len(tlist)-1, -1, -1):
        sortkey = get_sortkey(tlist[i])
        output[count_list[sortkey]-1] = tlist[i]
        count_list[sortkey] -=1

    return output
    
#----- take list from user-----------
li_st = [int(x) for x in input().split()]
print("Unsorted List")
print(li_st)

##------working on our algo-----------
print("\nSorted list using basic counting sort")
output = counting_sort(li_st, max(li_st) +1, get_sortkey) 
print(output)
