import time
import matplotlib.pyplot as plt

def collatz_sequence(n):
    """Generate the Collatz sequence for n."""
    steps = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps.append(n)
    return steps


def visualize(sequence, title="Collatz Sequence"):
    plt.clf()
    plt.plot(sequence, marker='o')
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.yscale("log")  # makes visualization MUCH nicer
    plt.grid(True)
    plt.pause(0.01)


def auto_mode(interval):
    print("\nAuto mode started.")
    print("Press SPACE in the plot window to stop.\n")

    plt.ion()
    stop = False

    def on_key(event):
        nonlocal stop
        if event.key == ' ':
            stop = True

    fig = plt.figure()
    fig.canvas.mpl_connect("key_press_event", on_key)

    n = 1
    while not stop:
        seq = collatz_sequence(n)
        visualize(seq, f"Collatz Sequence for n = {n}")
        n += 1
        time.sleep(interval)

    plt.ioff()
    plt.show()
    print("Auto mode stopped.")


# --- Main Program ---
try:
    num = int(input("Enter a positive integer (or -1 for auto mode): "))

    if num == -1:
        interval = float(input("Enter step interval time (seconds): "))
        auto_mode(interval)

    elif num <= 0:
        print("Please enter a positive number greater than 0.")

    else:
        seq = collatz_sequence(num)
        print("\nCollatz sequence:")
        for i, value in enumerate(seq, start=1):
            print(f"Step {i}: {value}")

        plt.ion()
        visualize(seq, f"Collatz Sequence for n = {num}")
        plt.ioff()
        plt.show()

except ValueError:
    print("Invalid input! Please enter a valid number.")
