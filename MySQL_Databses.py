import mysql.connector

# MySQl databses details

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="db_name"
)
mycursor = mydb.cursor()

# Execute SQL Query =>>>> mycursor.execute("SQL Query")
mycursor.execute("SELECT column FROM table")

myresult = mycursor.fetchall()

for x in myresult:
    print(x)
