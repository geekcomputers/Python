import pandas as pd

# reading csv file into python
df = pd.read_csv(
    r"c:\PROJECT\Drug_Recommendation_System\drug_recommendation_system\Drugs_Review_Datasets.csv"
)  # Replace the path with your own file path

print(df)

# Basic functions
print(df.info())  # Provides a short summary of the DataFrame
print(df.head())  # prints first 5 rows
print(df.tail())  # prints last 5 rows
print(df.describe())  # statistical summary of numeric columns
print(df.columns)  # Returns column names
print(df.shape)  # Returns the number of rows and columnsrr

print(
    help(pd)
)  # Use help(pd) to explore and understand the available functions and attributes in the pandas (pd) lib
