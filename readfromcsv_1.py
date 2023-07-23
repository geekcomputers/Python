__author__ = "vamsi"
import pandas as pd  # pandas library to read csv file
from matplotlib import pyplot as plt  # matplotlib library to visualise the data
from matplotlib import style

style.use("ggplot")

"""reading data from SalesData.csv file
    and passing data to dataframe"""

df = pd.read_csv("..\SalesData.csv")  # Reading the csv file
x = df[
    "SalesID"
].as_matrix()  # casting SalesID to list #extracting the column with name SalesID
y = df["ProductPrice"].as_matrix()  # casting ProductPrice to list
plt.xlabel("SalesID")  # assigning X-axis label
plt.ylabel("ProductPrice")  # assigning Y-axis label
plt.title("Sales Analysis")  # assigning Title to the graph
plt.plot(x, y)  # Plot X and Y axis
plt.show()  # Show the graph
