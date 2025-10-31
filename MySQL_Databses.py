import mysql.connector
# MySQl databses details
host = input("Enter MySQL host: ")
username = input("Enter MySQL username: ")
password = input("Enter MySQL password: ")
db_name = input("Enter MySQL database: ")
mydb = mysql.connector.connect(
    host=host, user=username, passwd=password, database=db_name
)
mycursor = mydb.cursor()
# Execute SQL Query =>>>> mycursor.execute("SQL Query")
mycursor.execute("SELECT column FROM table")

myresult = mycursor.fetchall()

for x in myresult:
    print(x)
