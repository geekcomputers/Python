
# This method is called exhaustive numeration!
# I am checking every possible value
# that can be root of given x systematically
# Kinda brute forcing

def find_cube_root(x):
    if type(x) == str:
        return "Expected an integer! Cannot find cube root of an string!"
    for i in range(0, x):
        if i ** 3 == x:
            return i 
    return "{} is not a perfect cube".format(x)
    
# Test 
x = 27
result = find_cube_root(x)
print("Cube root of {} is {}".format(x, result))
