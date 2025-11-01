# Laundry Service Class

## Overview
The LaundryService class is designed to manage customer details and calculate charges for a cloth and apparel cleaning service. It provides methods to create customer-specific instances, print customer details, calculate charges based on cloth type, branding, and season, and print final details including the expected day of return.

## Class Structure
### Methods
1. `__init__(name, contact, email, cloth_type, branded, season)`: Initializes a new customer instance with the provided details and assigns a unique customer ID.
   - Parameters:
     - `name`: String, name of the customer.
     - `contact`: Numeric (integer), contact number of the customer.
     - `email`: Alphanumeric (string), email address of the customer.
     - `cloth_type`: String, type of cloth deposited (Cotton, Silk, Woolen, or Polyester).
     - `branded`: Boolean (0 or 1), indicating whether the cloth is branded.
     - `season`: String, season when the cloth is deposited (Summer or Winter).
     
2. `customerDetails()`: Prints out the details of the customer, including name, contact number, email, cloth type, and whether the cloth is branded.

3. `calculateCharge()`: Calculates the charge based on the type of cloth, branding, and season.
   - Returns:
     - Numeric, total charge for cleaning the cloth.

4. `finalDetails()`: Calls `customerDetails()` and `calculateCharge()` methods within itself and prints the total charge and the expected day of return.
   - Prints:
     - Total charge in Rupees.
     - Expected day of return (4 days if total charge > 200, otherwise 7 days).

## Example Usage
```python
# Example usage:
name = input("Enter customer name: ")
contact = int(input("Enter contact number: "))
email = input("Enter email address: ")
cloth_type = input("Enter cloth type (Cotton/Silk/Woolen/Polyester): ")
branded = bool(int(input("Is the cloth branded? (Enter 0 for No, 1 for Yes): ")))
season = input("Enter season (Summer/Winter): ")

customer = LaundryService(name, contact, email, cloth_type, branded, season)
customer.finalDetails()


markdown
Copy code
# Laundry Service Class

## Overview
The LaundryService class is designed to manage customer details and calculate charges for a cloth and apparel cleaning service. It provides methods to create customer-specific instances, print customer details, calculate charges based on cloth type, branding, and season, and print final details including the expected day of return.

## Class Structure
### Methods
1. `__init__(name, contact, email, cloth_type, branded, season)`: Initializes a new customer instance with the provided details and assigns a unique customer ID.
   - Parameters:
     - `name`: String, name of the customer.
     - `contact`: Numeric (integer), contact number of the customer.
     - `email`: Alphanumeric (string), email address of the customer.
     - `cloth_type`: String, type of cloth deposited (Cotton, Silk, Woolen, or Polyester).
     - `branded`: Boolean (0 or 1), indicating whether the cloth is branded.
     - `season`: String, season when the cloth is deposited (Summer or Winter).
     
2. `customerDetails()`: Prints out the details of the customer, including name, contact number, email, cloth type, and whether the cloth is branded.

3. `calculateCharge()`: Calculates the charge based on the type of cloth, branding, and season.
   - Returns:
     - Numeric, total charge for cleaning the cloth.

4. `finalDetails()`: Calls `customerDetails()` and `calculateCharge()` methods within itself and prints the total charge and the expected day of return.
   - Prints:
     - Total charge in Rupees.
     - Expected day of return (4 days if total charge > 200, otherwise 7 days).

## Example Usage
```python
# Example usage:
name = input("Enter customer name: ")
contact = int(input("Enter contact number: "))
email = input("Enter email address: ")
cloth_type = input("Enter cloth type (Cotton/Silk/Woolen/Polyester): ")
branded = bool(int(input("Is the cloth branded? (Enter 0 for No, 1 for Yes): ")))
season = input("Enter season (Summer/Winter): ")

customer = LaundryService(name, contact, email, cloth_type, branded, season)
customer.finalDetails()
Usage Instructions
Create an instance of the LaundryService class by providing customer details as parameters to the constructor.
Use the finalDetails() method to print the customer details along with the calculated charge and expected day of return.


Contributors
(Rohit Raj)[https://github.com/MrCodYrohit]


