'''Author Anurag Kumar(mailto:anuragkumara95@gmail.com)

A prime number is a natural number that has exactly two distinct natural number divisors: 1 and itself.

#USAGE:
  - $pythonfind_prime.py <num:int>

##THEORY
-Sieve of Eratosthenes(source:wikipedia.com)
    In mathematics, the sieve of Eratosthenes is a simple, ancient algorithm for finding all prime numbers up to any given limit.

    It does so by iteratively marking as composite (i.e., not prime) the multiples of each prime, starting with the first prime 
    number, 2. The multiples of a given prime are generated as a sequence of numbers starting from that prime, with constant 
    difference between them that is equal to that prime. This is the sieve's key distinction from using trial division to 
    sequentially test each candidate number for divisibility by each prime.

    To find all the prime numbers less than or equal to a given integer n by Eratosthenes' method:

      - Create a list of consecutive integers from 2 through n: (2, 3, 4, ..., n).
      - Initially, let p equal 2, the smallest prime number.
      - Enumerate the multiples of p by counting to n from 2p in increments of p, and mark them in the list (these will be 2p, 
        3p, 4p, ...; the p itself should not be marked).
      - Find the first number greater than p in the list that is not marked. If there was no such number, stop. Otherwise, let 
        p now equal this new number (which is the next prime), and repeat from step 3.
      - When the algorithm terminates, the numbers remaining not marked in the list are all the primes below n.
'''
import sys


def find_prime(num):
    res_list = []
    for i in range(2, num + 1):
        if res_list != [] and any(i % l == 0 for l in res_list):
            continue
        res_list.append(i)
    return res_list


if __name__ == "__main__":
    if len(sys.argv) != 2: raise Exception("usage - $python find_prime.py <num:int>")
    try:
        num = int(sys.argv[1])
    except ValueError:
        raise Exception("Enter an integer as argument only.")
    l = find_prime(num)
    print(l)
