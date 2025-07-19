num = 11
# Negative numbers, 0 and 1 are not primes
if num > 1:
    # Iterate from 2 to n // 2
    for i in range(2, (num // 2) + 1):
        # If num is divisible by any number between
        # 2 and n / 2, it is not prime
        if (num % i) == 0:
            print(num, "is not a prime number")
            break
    else:
        print(num, "is a prime number")
else:
    print(num, "is not a prime number")
