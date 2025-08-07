import pandas as pd

# loading the dataset

df = pd.read_csv(
    r"c:\PROJECT\Drug_Recommendation_System\drug_recommendation_system\Drugs_Review_Datasets.csv"
)

print(df)  # prints Dataset
# funtions
print(df.tail())
print(df.head())
print(df.info())
print(df.describe())
print(df.column)
print(df.shape())
