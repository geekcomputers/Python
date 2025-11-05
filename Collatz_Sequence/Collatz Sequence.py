def collatz_sequence(n):
    """Generate and print the Collatz sequence for n."""
    steps = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps.append(n)
    return steps


# --- Main Program ---
try:
    num = int(input("Enter a positive integer: "))
    if num <= 0:
        print("Please enter a positive number greater than 0.")
    else:
        sequence = collatz_sequence(num)
        print("\nCollatz sequence:")
        for i, value in enumerate(sequence, start=1):
            print(f"Step {i}: {value}")
except ValueError:
    print("Invalid input! Please enter an integer.")
