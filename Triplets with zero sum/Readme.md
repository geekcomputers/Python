Problem : **Given an array of distinct elements. The task is to find triplets in the array whose sum is zero.**

Method : This method uses Sorting to arrive at the correct result and is solved in O(n^2) time.

Approach: 
The above method requires extra space. The idea is based on method 2 of this post. For every element check that there is a pair whose sum is equal to the negative value of that element.
Algorithm:
1. Sort the array in ascending order.
2. Traverse the array from start to end.
3. For every index i, create two variables l = i + 1 and r = n – 1
4. Run a loop until l is less than r, if the sum of array[l], array[r] is equal to zero then print the triplet and break the loop
5. If the sum is less than zero then increment value of l, by increasing value of l the sum will increase as the array is sorted, so array[l+1] > array [l]
6. If the sum is greater than zero then decrement value of r, by increasing value of l the sum will decrease as the array is sorted, so array[r-1] < array [r].
