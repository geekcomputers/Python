# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Housing Price Predictor

import pandas as pd

housing = pd.read_csv("data.csv")

housing.head()

housing.info()

housing.describe()

# %matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20, 15))

# # Train-Test Splitting

# +
import numpy as np


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# -

train_set, test_set = split_train_test(housing, 0.2)

print(f"Rows in train set: {len(train_set)} \nRows in test set : {len(test_set)}")

# +
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)} \nRows in test set : {len(test_set)}")

# +
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["CHAS"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# -

strat_test_set.describe()

strat_test_set["CHAS"].value_counts()

strat_train_set["CHAS"].value_counts()

housing = strat_train_set.copy()  # use just after split data

# ## Looking for Correlations

corr_matrix = housing.corr()

corr_matrix["MEDV"].sort_values(ascending=False)

# +
from pandas.plotting import scatter_matrix

attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# -

housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)

housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# ## Missing  Attributes

# + active=""
# To take care of missing attributes, you have 3 options
#  1. get Rid of the missing data points
#     a=housing.dropna(subset=["RM"])
#     a.shape
#  2. Get rid of the whole attribute
#      housing.drop("RM", axis=1)
#  3. Set the value to some value(0,mean or medium)
#     #median=housing["RM"].median()
#     #housing["RM"].fillna(median)
#     #housing.shape
# -

median = housing["RM"].median()
housing["RM"].fillna(median)
housing.shape

# +
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(housing)
# -

imputer.statistics_.shape
imputer.statistics_

X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()

# ## Scikit-learn Design 

# Basically, there are 3 types of objects:
# 1. Estimators - it estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method.Fit method -Fits the dataset and calculates internal parameters
#
# 2. Transformers - transform method takes input and returns output based on the learning from fit(). It also has a convenience function called fit_transform() which fits and then transforms.
#
# 3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.

# ## Feature Scaling

# Primarily, two types of features scaling methods:
# 1. Min-max scaling (Normalization)
#    0 < (value-min)/(max-min) >1
#    Sklearn provides a class called MinMaxScaler for this
#    
# 2. Standardization
#    (value-mean)/std
#    Sklearn provides a class called StandardScaler for this

# +
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("std_scaler", StandardScaler())]
)
# -

housing_num_tr = my_pipeline.fit_transform(housing)

housing_num_tr.shape

# ## Selecting a desired model

# +
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)
# -

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

list(some_labels)

# +
from sklearn.metrics import mean_squared_error

housing_predictions = model.predict(housing_num_tr)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# -

lin_mse

lin_rmse

# ## Cross Validation

# +
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10
)
rmse_scores = np.sqrt(-scores)
# -

rmse_scores


def print_scores(scores):
    print("scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


print_scores(rmse_scores)

# ## Saving Model 

# +
from joblib import dump, load

dump(model, "HousingPricePredicter.joblib")
# -

# ## Testing the model on test data 

X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse
