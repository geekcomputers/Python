# counting the number of occurrences of a letter in a string using defaultdict
# left space in starting for clarity
from collections import defaultdict

s = "mississippi"
d = defaultdict(int)
for k in s:
    d[k] += 1
sorted(d.items())
print(d)

# OUTPUT --- [('i', 4), ('m', 1), ('p', 2), ('s', 4)]
