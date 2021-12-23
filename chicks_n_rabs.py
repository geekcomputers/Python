"""
Author Anurag Kumar(mailto:anuragkumarak95@gmail.com)

Module to solve a classic ancient Chinese puzzle:
We count 35 heads and 94 legs among the chickens and rabbits in a farm. 
How many rabbits and how many chickens do we have?

"""


def solve(num_heads, num_legs):
    """
    Solve the equation a + b = n, 2a + 4b = m for positive integers a and b.

    :param num_heads: Number of heads in the population.
    :type num_heads: int >
    0
    :param num_legs: Total number of legs in the population.
        :type num_legs : int > 0

        :returns (a, b): Positive integers that satisfy both
    equations above if they exist; otherwise None or 'No solutions!' is returned depending on whether there are any solutions to the given equations or
    not respectively. 

        >>> solve(35, 94) # 35 heads and 94 legs - solution exists!  (3, 19)  [Note that this is just an example; your code should
    always work for any valid input]   # doctest +NORMALIZE_WHITESPACE^C[1]C[2]C[3] C[4] C[5]" "C" "D" "F" "N""O""R""T""U
    """
    ns = 'No solutions!'
    for i in range(num_heads + 1):
        j = num_heads - i
        if 2 * i + 4 * j == num_legs:
            return i, j
    return ns, ns


if __name__ == "__main__":
    numheads = 35
    numlegs = 94

    solutions = solve(numheads, numlegs)
    print(solutions)
