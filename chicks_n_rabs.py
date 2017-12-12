'''Author Anurag Kumar(mailto:anuragkumarak95@gmail.com)

Module to solve a classic ancient Chinese puzzle:
We count 35 heads and 94 legs among the chickens and rabbits in a farm. 
How many rabbits and how many chickens do we have?


'''
def solve(numheads,numlegs):
    ns='No solutions!'
    for i in range(numheads+1):
        j=numheads-i
        if 2*i+4*j==numlegs:
            return i,j
    return ns,ns

if __name__=="__main__":
    numheads=35
    numlegs=94

    solutions=solve(numheads,numlegs)
    print(solutions)
