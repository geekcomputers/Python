__author__ = 'vamsi'
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

"""reading data from SalesData.csv file
    and passing data to dataframe"""

df=pd.read_csv("C:\\Users\\Test\\Desktop\\SalesData.csv")
x=df["SalesID"].tolist()#casting SalesID to list
y=df["ProductPrice"].tolist()#casting ProductPrice to list
plt.xlabel("SalesID")#assigning X-axis label
plt.ylabel("ProductPrice")#assigning Y-axis label
plt.title("Sales Analysis")#assigning Title to the graph
plt.plot(x,y)#Plot X and Y axis
plt.show()#Show the graph