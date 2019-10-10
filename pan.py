import numpy as np
import pandas as pd
from matplotlib import *

# .........................Series.......................#

x1 = np.array([1, 2, 3, 4])
s = pd.Series(x1, index=[1, 2, 3, 4])
print(s)

# .......................DataFrame......................#

x2 = np.array([1, 2, 3, 4, 5, 6])
s = pd.DataFrame(x2)
print(s)

x3 = np.array([['Alex', 10], ['Nishit', 21], ['Aman', 22]])
s = pd.DataFrame(x3, columns=['Name', 'Age'])
print(s)

data = {'Name': ['Tom', 'Jack', 'Steve', 'Ricky'], 'Age': [28, 34, 29, 42]}
df = pd.DataFrame(data, index=['rank1', 'rank2', 'rank3', 'rank4'])
print(df)

data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4, 'c': 5}]
df = pd.DataFrame(data)
print(df)

d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print(df)

# ....Adding New column......#

data = {'one': pd.Series([1, 2, 3, 4], index=[1, 2, 3, 4]),
        'two': pd.Series([1, 2, 3], index=[1, 2, 3])}
df = pd.DataFrame(data)
print(df)
df['three'] = pd.Series([1, 2], index=[1, 2])
print(df)

# ......Deleting a column......#

data = {'one': pd.Series([1, 2, 3, 4], index=[1, 2, 3, 4]),
        'two': pd.Series([1, 2, 3], index=[1, 2, 3]),
        'three': pd.Series([1, 1], index=[1, 2])
        }
df = pd.DataFrame(data)
print(df)
del df['one']
print(df)
df.pop('two')
print(df)

# ......Selecting a particular Row............#

data = {'one': pd.Series([1, 2, 3, 4], index=[1, 2, 3, 4]),
        'two': pd.Series([1, 2, 3], index=[1, 2, 3]),
        'three': pd.Series([1, 1], index=[1, 2])
        }
df = pd.DataFrame(data)
print(df.loc[2])
print(df[1:4])

# .........Addition of Row.................#

df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['a', 'b'])

df = df.append(df2)
print(df.head())

# ........Deleting a Row..................#

df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['a', 'b'])

df = df.append(df2)

# Drop rows with label 0
df = df.drop(0)

print(df)

# ..........................Functions.....................................#


d = {'Name': pd.Series(['Tom', 'James', 'Ricky', 'Vin', 'Steve', 'Smith', 'Jack']),
     'Age': pd.Series([25, 26, 25, 23, 30, 29, 23]),
     'Rating': pd.Series([4.23, 3.24, 3.98, 2.56, 3.20, 4.6, 3.8])}

df = pd.DataFrame(d)
print("The transpose of the data series is:")
print(df.T)
print(df.shape)
print(df.size)
print(df.values)

# .........................Statistics.......................................#

d = {'Name': pd.Series(['Tom', 'James', 'Ricky', 'Vin', 'Steve', 'Smith', 'Jack',
                        'Lee', 'David', 'Gasper', 'Betina', 'Andres']),
     'Age': pd.Series([25, 26, 25, 23, 30, 29, 23, 34, 40, 30, 51, 46]),
     'Rating': pd.Series([4.23, 3.24, 3.98, 2.56, 3.20, 4.6, 3.8, 3.78, 2.98, 4.80, 4.10, 3.65])
     }
df = pd.DataFrame(d)
print(df.sum())

d = {'Name': pd.Series(['Tom', 'James', 'Ricky', 'Vin', 'Steve', 'Smith', 'Jack',
                        'Lee', 'David', 'Gasper', 'Betina', 'Andres']),
     'Age': pd.Series([25, 26, 25, 23, 30, 29, 23, 34, 40, 30, 51, 46]),
     'Rating': pd.Series([4.23, 3.24, 3.98, 2.56, 3.20, 4.6, 3.8, 3.78, 2.98, 4.80, 4.10, 3.65])
     }
df = pd.DataFrame(d)
print(df.describe(include='all'))

# .......................Sorting..........................................#

# Using the sort_index() method, by passing the axis arguments and the order of sorting,
# DataFrame can be sorted. By default, sorting is done on row labels in ascending order.

unsorted_df = pd.DataFrame(np.random.randn(10, 2), index=[1, 4, 6, 2, 3, 5, 9, 8, 0, 7], columns=['col2', 'col1'])

sorted_df = unsorted_df.sort_index()
print(sorted_df)
sorted_df = unsorted_df.sort_index(ascending=False)
print(sorted_df)

# By passing the axis argument with a value 0 or 1,
# the sorting can be done on the column labels. By default, axis=0, sort by row.
# Let us consider the following example to understand the same.

unsorted_df = pd.DataFrame(np.random.randn(10, 2), index=[1, 4, 6, 2, 3, 5, 9, 8, 0, 7], columns=['col2', 'col1'])
sorted_df = unsorted_df.sort_index(axis=1)
print(sorted_df)

unsorted_df = pd.DataFrame({'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4]})
sorted_df = unsorted_df.sort_values(by='col1', kind='mergesort')

# print (sorted_df)

# ...........................SLICING...............................#

df = pd.DataFrame(np.random.randn(8, 4),
                  index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], columns=['A', 'B', 'C', 'D'])
# Select all rows for multiple columns, say list[]
print(df.loc[:, ['A', 'C']])
print(df.loc[['a', 'b', 'f', 'h'], ['A', 'C']])

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
# Index slicing
print(df.ix[:, 'A'])

# ............................statistics......................#

s = pd.Series([1, 2, 3, 4, 5, 4])
print(s.pct_change())

df = pd.DataFrame(np.random.randn(5, 2))
print(df.pct_change())

df = pd.DataFrame(np.random.randn(10, 4),
                  index=pd.date_range('1/1/2000', periods=10),
                  columns=['A', 'B', 'C', 'D'])
print(df.rolling(window=3).mean())

print(df.expanding(min_periods=3).mean())

# ........................MISSING DATA............................................#

df = pd.DataFrame(np.random.randn(3, 3), index=['a', 'c', 'e'], columns=['one',
                                                                         'two', 'three'])

df = df.reindex(['a', 'b', 'c'])

print(df)
print("NaN replaced with '0':")
print(df.fillna(0))

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
                                                'h'], columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

print(df)
print(df.fillna(method='pad'))
print(df.fillna(method='bfill'))
print(df.dropna())
print(df.dropna(axis=1))

# .........................Grouping...............................................#

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
                     'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
            'Rank': [1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 2],
            'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
            'Points': [876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]}
df = pd.DataFrame(ipl_data)

grouped = df.groupby('Year')

for name, group in grouped:
    print(name)
    print(group)

print(grouped.get_group(2014))
grouped = df.groupby('Team')
print(grouped['Points'].agg([np.sum, np.mean, np.std]))

# ...............................Reading a Csv File............................#

data = pd.read_csv("dat.csv")
print(data)
