"""Time Delta Solution """


# -----------------------------------------------------------------------------
# You are givent two timestams in the format: Day dd Mon yyyy hh:mm:ss +xxxx
# where +xxxx represents the timezone.

# Input Format:
# The first line contains T, the number of test cases.
# Each test case contains two lines, representing the t1 and t2 timestamps.

# Constraints:
# input contains only valid timestamps.
# year is  < 3000.

# Output Format:
# Print the absoulte diffrence (t2 - t1) in seconds.

# Sample Input:
# 2
# Sun 10 May 2015 13:54:36 -0700
# Sun 10 May 2015 13:54:36 -0000
# Sat 02 May 2015 19:54:36 +0530
# Fri 01 May 2015 13:54:36 -0000

# Sample Output:
# 25200
# 88200
#------------------------------------------------------------------------------

# Imports
import math
import os 
import random
import re
import sys
import datetime

# Complete the time_delta function below.
def time_delta(t1, t2):
    """
    Calculate the time delta between two timestamps in seconds.
    """
    # Convert the timestamps to datetime objects
    t1 = datetime.datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    t2 = datetime.datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')

    return (t1 - t2)



if __name__ == '__main__':

    t = int(input())

    for itr_t in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)
        # print Delta with 1 Decimal Place
        print(round(delta.total_seconds(), 1))




