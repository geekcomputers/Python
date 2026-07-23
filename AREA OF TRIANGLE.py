def get_valid_side(prompt:str):
  while True:
    try:
      value = float(input(prompt))
      if value <=0:
        print("Side must be positive")
        continue
      return value
    except ValueError:
      print("Invalid Input")


a = get_valid_side("Enter side 1: ")
b = get_valid_side("Enter side 2: ")
c = get_valid_side("Enter side 3: ")

semi_perimeter = (a + b + c) / 2

area = sqrt((s * (s - a) * (s - b) * (s - c)))
print("The area of the triangle is %0.2f" % area)
