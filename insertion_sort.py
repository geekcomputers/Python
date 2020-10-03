#insertion sort

list = []  # declaring list


def input_list():
    #taking length and then values of list as input from user
    n = int(input("Enter number of elements in the list: "))  # taking value from user
    for i in range(n):
        temp = int(input("Enter element " + str(i + 1) + ': '))
        list.append( temp )


def insertion_sort(list,n):
    """
    sort list in assending order

    INPUT: 
        list=list of values to be sorted
        n=size of list that contains values to be sorted

    OUTPUT:
        list of sorted values in assending order
    """
    for i in range(0,n):
        key = list[i]
        j = i - 1
        #Swap elements witth key iff they are
        #greater than key
        while j >= 0 and list[j] > key:
            list[j + 1] = list[j]
            j = j - 1
        list[j + 1] = key 
    return list


def insertion_sort_desc(list,n):
    """
    sort list in desending order

    INPUT: 
        list=list of values to be sorted
        n=size of list that contains values to be sorted

    OUTPUT:
        list of sorted values in desending order
    """
    for i in range(0,n):
        key = list[i]
        j = i - 1
        #Swap elements witth key iff they are
        #greater than key
        while j >= 0 and list[j] < key:
            list[j + 1] = list[j]
            j = j - 1
        list[j + 1] = key
    return list  


input_list()
list1=insertion_sort(list,len(list))
print(list1)
list2=insertion_sort_desc(list,len(list))
print(list2)