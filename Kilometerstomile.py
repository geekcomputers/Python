def kilometers_to_miles(kilometers):
    """
    Convert kilometers to miles using the conversion factor.
    Params:
        kilometers: float
    Return:
        float
    """
    conv_fac = 0.621371 
    return kilometers * conv_fac


if __name__ == "__main__":
    # Taking kilometers input from the user
    kilometers = float(input("Enter value in kilometers: "))
    miles = kilometers_to_miles(kilometers)
    print(f"{kilometers:.2f} kilometers is equal to {miles:.2f} miles")
