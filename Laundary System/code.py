id=1
class LaundryService:
  def __init__(self,Name_of_customer,Contact_of_customer,Email,Type_of_cloth,Branded,Season,id):
    self.Name_of_customer=Name_of_customer
    self.Contact_of_customer=Contact_of_customer
    self.Email=Email
    self.Type_of_cloth=Type_of_cloth
    self.Branded=Branded
    self.Season=Season
    self.id=id

  def customerDetails(self):
    print("The Specific Details of customer:")
    print("customer ID: ",self.id)
    print("customer name:", self.Name_of_customer)
    print("customer contact no. :", self.Contact_of_customer)
    print("customer email:", self.Email)
    print("type of cloth", self.Type_of_cloth)
    if self.Branded == 1:
      a=True
    else:
      a=False
    print("Branded", a)
  def calculateCharge(self):
    a=0
    if self.Type_of_cloth=="Cotton":
      a=50.0
    elif self.Type_of_cloth=="Silk":
      a=30.0
    elif self.Type_of_cloth=="Woolen":
      a=90.0
    elif self.Type_of_cloth=="Polyester":
      a=20.0
    if self.Branded==1:
      a=1.5*(a)
    else:
      pass
    if self.Season=="Winter":
      a=0.5*a
    else:
      a=2*a
    print(a)
    return a
  def finalDetails(self):
    self.customerDetails()
    print("Final charge:",end="")
    if self.calculateCharge() >200:
      print("to be return in 4 days")
    else:
      print("to be return in 7 days")
while True:
  name=input("Enter the name: ")
  contact=int(input("Enter the contact: "))
  email=input("Enter the email: ")
  cloth=input("Enter the type of cloth: ")
  brand=bool(input("Branded ? "))
  season=input("Enter the season: ")
  obj=LaundryService(name,contact,email,cloth,brand,season,id)
  obj.finalDetails()
  id=id+1
  z=input("Do you want to continue(Y/N):")
  if z=="Y":
    continue
  elif z =="N":
    print("Thanks for visiting!")
    break
  else:
    print("Select valid option")






      