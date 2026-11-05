#! /usr/bin/python3

"""
Author:
    Name: Ayush Gupta
    Email: in.ayushgupta@gmail.com
    GitHub: inayushgupta

Description:
The Fibonacci code avoid the extra work.
It not only utlizes dynamic programming and
avoid repreated calls for same calculations
but also it does saves that result for multiple
values.
Sorted list, for example [4, 9, 10] does not
have to calculate fibonacci for every numebr
-- 4 : 1, 2, 3, 4
-- 9 : 1, 2, 3, 4, 5, 6, 7, 8, 9
-- 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

It saves the result as it goes and avoid recalculation

-- 4 : 1, 2, 3, 4 --> saves 3 and 4
-- 9 : ( use 3 and 4 ) --> 5, 6, 7, 8, 9 --> saves 8 and 9
-- 10: ( use 8 and 9 ) --> 10

saving so much work.
"""

import sys

# global state tracking
_last = 1
_prev, _curr = 0, 1

# calculate the next fib using saved states
def fib(n):
    global _last, _prev, _curr

    if n in (0, 1):
        return n

    for _ in range(_last, n):
        _curr, _prev = _prev + _curr, _curr
    
    _last = n

    return _curr 
    

def interactive_input():

    tries = 3
    while tries:
        n = input("Enter your number: ")
        if n.isdigit():
            return int(n)
        print("INVALID INPUT: Please enter a valid number.")
        tries -= 1
    print("MAXIMUM NUMBER OF TRIES: exiting")
    sys.exit(1)


def process_args() -> list[int]:

    args = sys.argv

    if len(args) < 2:
        number = interactive_input()
        if not number:
            sys.exit(1)
        return [number]
   
    nums = []
    for i in range(1, len(args)):
        if not args[i].isdigit():
            print("INVALID INPUT: one or more input numbers are not valid.")
            sys.exit(1)
        nums.append(int(args[i]))


    return nums

def main():
   
    nums = process_args()

    # sort the number to cut some time off
    nums.sort()

    # remove duplicates
    sorted_nums = [nums[0]]
    prev = nums[0]
    for num in nums[1:]:
        if num == prev:
            continue
        sorted_nums.append(num)
        prev = num

    for num in sorted_nums:
        print(f"{num}:{fib(num)}")

    sys.exit(0)

main()
