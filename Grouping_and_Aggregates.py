import pandas as pd
import numpy as np

# Loading the shopping dataset.
shopping_df = pd.read_csv('shopping_data.csv')

# Loading the film dataset.
film_df = pd.read_csv('film.csv')

""" 
Grouping:-

pd.DataFrame.group_by:-
--> pd.DataFrame.group_by(by=None, axis=0, sort=True, dropna=True)
    1. 'by' is used to determine the groups.
    2. 'axis' specifies whether to split across rows or columns.
"""
print(shopping_df)
product_grp = shopping_df.groupby(['Product'])
print(product_grp)

print(product_grp.groups)
print(product_grp.get_group('Macbook Pro Laptop'))
print(type(product_grp.get_group('Macbook Pro Laptop')))

print(film_df)
films_group = film_df.groupby(['Subject','Year'])
print(films_group)
print(films_group.groups)

""" 
--> By default 'dropna' is True, so group keys containing 'NaN' values are dropped along with the row/column.
"""

values = [[1,2,3],[1,None,4],[2,1,3],[1,2,2,]]
numeric_df = pd.DataFrame(values, columns=["a","b","c"])
print(numeric_df)

group_b = numeric_df.groupby(by=["b"])
print(group_b.groups)

""" 
Aggegations:-
    1. Aggregation operations are always performed over an axis, either the index (default) or the column axis.
    2. This behaviour is different from numpy aggregation functions (mean, meadian, prod, sum, std, var), where the default is to compute the aggregation of flattened arrays, e.g., numpy.mean(arr_2d) as opposed to numpy.mean(arr_2d, axis=0).
"""
print(numeric_df)
numeric_df_mean = numeric_df.mean()
print(numeric_df_mean)

print("mean of prices:", shopping_df['Price Each'].mean())
print("median of prices:", shopping_df['Price Each'].median())
print("maximum of prices:", shopping_df['Price Each'].max())
print("total number of products:",shopping_df["Product"].count())

""" 
pd.DataFrame.value_counts:-
--> Can be used to count the number of times each value is repeated.
"""

print(shopping_df['Product'].unique())

print('Count of each distinct product:\n', shopping_df['Product'].value_counts())

""" 
--> If 'normalize=True' then the object returned will contain the relative frequencies of the unique values.
"""

print("Percentages (normalizations) of unique values:\n",shopping_df['Product'].value_counts(normalize=True))

# Applying aggregations on groups:-
print(shopping_df)

iphone_filter = (shopping_df['Product'] == "iPhone")
print(shopping_df[iphone_filter]["Price Each"].median())

print(shopping_df.groupby('Product')["Price Each"].median())

print(shopping_df.groupby('Product').median())

""" 
Effectively, 'shopping_df' is
    1. Split into groups based on 'Product'.
    2. 'median' function is applied to each group.
    3. The results from reach group are combined into a 'DataFrame'.
"""
print(film_df)

print(film_df.groupby('Director')['Popularity'].mean())

print(film_df.groupby('Director')['Popularity'].mean().loc['Abrahams, Jim'])

# Using aggregations with filters:-

print(shopping_df)

filt = (shopping_df['Product'] == "Flatscreen TV")
print(shopping_df.loc[filt]['Purchase Address'].str.contains('CA'))

# Here, 'sum' will give the number of True values.

print(shopping_df.loc[filt]['Purchase Address'].str.contains('CA').sum())

""" 
pd.DataFrame.aggregate:-
--> pd.DataFrame.aggregate(func=None, axis=0, *args, **kwargs)
    1. 'func' is the functon to use for aggregating the data.
    2. 'axis' specifies whether to apply the function to each row or each column.
    3. '*args' are the positional arguments to pass to func.
    4. '**kwargs' are the keyword arguments to pass to func.
    5. 'agg' is an alias for aggregate.
"""

df = pd.DataFrame([[1,2,3],
                   [4,5,6],
                   [7,8,9]],
                  columns=['A','B','C'])
print(df)
print(df.mean())
print(df.agg(['mean']))
print(df.agg(['sum','min','mean']))

year_grp = film_df.groupby('Year')
print(year_grp['Popularity'].agg(['median','mean']))

print(shopping_df)

product_grp = shopping_df.groupby('Product')
print(product_grp['Price Each'].agg(['median','mean']).loc['Flatscreen TV'])

filt = (shopping_df['Product'] == 'Flatscreen TV')
print(shopping_df[filt]['Purchase Address'].str.contains('CA').sum())

""" 
--> The following code throws an error because 'product_grp['Purchase Address']' is not a 'Series' object.
--> It is a 'SeriesGroupBy' object.
"""

product_grp = shopping_df.groupby('Product')
# print(product_grp['Purchase Address'].str.contains('CA').sum())

print(type(product_grp['Purchase Address']))

# Use the '.apply' method on 'SeriesGroupBy' objects.

print(product_grp['Purchase Address'].apply(lambda x: x.str.contains('CA').sum()))

""" 
pd.DataFrame.cumsum:-
--> pd.DataFrame.cumsum(axis=None, skipna=True)
    1. Returns the cumulative sum of a 'Series' or 'DataFrame'.

--> Series:-
    2. By default, 'NaN' values are ignored.
"""

series = pd.Series([3,np.nan,4,-6,0])
print(series)
print(series.cumsum())

""" 
Dataframe:-
    1. By default, it iterates over the rows and finds the sum in each column.
    2. This is equivalent to 'axis=None' or 'axis='index''.
"""

df = pd.DataFrame({
    'A':[1.0,-3.0,2.0],
    'B':[1.0,np.nan,0.0],
    'C':[3.0,-2.0,-1.1]
})
print(df)
print(df.cumsum())

# To iterate over the columns and find the sum in each row,use 'axis=1'.
print(df.cumsum(axis=1))