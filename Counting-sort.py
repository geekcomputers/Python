#python program for counting sort (updated)
n=int(input("please give the number of elements\n"))
print("okey now plase enter n numbers seperated by spaces")
tlist=list(map(int,input().split()))
k = max(tlist)
n = len(tlist)


def counting_sort(tlist, k, n):
 
    """ Counting sort algo with sort in place.
        Args: 
            tlist: target list to sort
            k: max value assume known before hand
            n: the length of the given list
            map info to index of the count list.
        Adv:
            The count (after cum sum) will hold the actual position of the element in sorted order
            Using the above, 
            
    """

    # Create a count list and using the index to map to the integer in tlist.
    count_list = [0]*(k+1)

    # iterate the tgt_list to put into count list
    for i in range(0,n):
        count_list[tlist[i]] += 1  
        
    # Modify count list such that each index of count list is the combined sum of the previous counts 
    # each index indicate the actual position (or sequence) in the output sequence.
    for i in range(1,k+1):
        count_list[i] = count_list[i] + count_list[i-1]
 
    flist = [0]*(n)
    for i in range(n-1,-1,-1):
      count_list[tlist[i]] =count_list[tlist[i]]-1 
      flist[count_list[tlist[i]]]=(tlist[i])

    return flist

flist = counting_sort(tlist, k, n)
print(flist)

