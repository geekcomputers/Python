import random


def get_die_art(value):
  """Returns a list of 5 strings representing the box-drawing art

  for a given die face value (1-6).
  """
  faces = {
      1: ["┌───────┐", "│       │", "│   ●   │", "│       │", "└───────┘"],
      2: ["┌───────┐", "│ ●     │", "│       │", "│     ● │", "└───────┘"],
      3: ["┌───────┐", "│ ●     │", "│   ●   │", "│     ● │", "└───────┘"],
      4: ["┌───────┐", "│ ●   ● │", "│       │", "│ ●   ● │", "└───────┘"],
      5: ["┌───────┐", "│ ●   ● │", "│   ●   │", "│ ●   ● │", "└───────┘"],
      6: ["┌───────┐", "│ ●   ● │", "│ ●   ● │", "│ ●   ● │", "└───────┘"],
  }
  return faces.get(value, faces[1])


def main():
  print("========================================")
  print("       Interactive Dice Roller          ")
  print("========================================")

  # Get valid user input for the number of dice
  while True:
    try:
      num_dice = int(input("How many dice would you like to roll? (1-10): "))
      if 1 <= num_dice <= 10:
        break
      print("Please choose a number between 1 and 10.")
    except ValueError:
      print("Invalid input. Please enter a valid integer.")

  # Generate random values for each die
  rolls = [random.randint(1, 6) for _ in range(num_dice)]

  print(f"\nRolling {num_dice} dice...")

  # Retrieve ASCII art for each rolled die
  dice_arts = [get_die_art(val) for val in rolls]

  separator = "  "
  die_width = len(dice_arts[0][0])
  divider = "~" * (num_dice * die_width + (num_dice - 1) * len(separator))
  print(divider)

  # Print dice side-by-side by iterating line by line (0 to 4)
  for line_index in range(5):
    row_line = separator.join(dice_art[line_index] for dice_art in dice_arts)
    print(row_line)

  print(divider)

  # Output results and total sum
  print(f"Individual Rolls: {rolls}")
  print(f"Total Sum: {sum(rolls)}")
  print("========================================")


if __name__ == "__main__":
  main()
