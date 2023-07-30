# Taking kilometers input from the user
kilometers = float(input("Enter value in kilometers: "))

# conversion factor
conv_fac = 0.621371

# calculate miles
miles = kilometers * conv_fac
print(f'{kilometers:.2f} kilometers is equal to {miles:.2f} miles')
