import pandas as pd

# Loading dataset
shopping_df = pd.read_csv('shopping_data_v2.csv')

# Display Options:-

# print(shopping_df) # here it is default displaying top 5 and bottom 5 rows (total_rows = 10).

# display 8 rows (top = 4; bottom =4)

pd.set_option('display.max_rows',8)
print(shopping_df)

# To display more than 10 rows when the dataframe is truncated, set the 'min_rows' option to be greater than 10.

pd.set_option('display.min_rows',25)
pd.set_option('display.max_rows',30)
print(shopping_df)

pd.set_option('display.max_rows',2)
print(shopping_df)

# To display all the rows,set 'max_rows' to None.

pd.set_option('display.max_rows',None)    # Here it displays all the rows and columns of the given dataset.
print(shopping_df)

# To reset all the display options.

pd.reset_option('display')
print(shopping_df)

""" 
Index:-
    1. An index contains identifiers for rows. These are usually unique, but pandas doesn't enforce the uniqueness.
    2. Each identifier is also called a label.
"""

print(shopping_df.index)
print(shopping_df.index.name)

""" 
Changing the Index:-

pd.DataFrame.set_index:-
--> pd.DataFrame.set_index(keys, inplace=False)
    1. Set the DataFrame index (row labels) using one or more existing columns or arrays (of appropriate length).
    2. The index can replace the existing index or expand on it.
"""
print(shopping_df.columns)

setting_index_order_id = shopping_df.set_index('Order ID')  # Here we are setting the order ID as index of the tables dataset.
print(setting_index_order_id)

# Create a Multi-Index using the columns 'Order ID' and 'Product' :

multi_indexing = shopping_df.set_index(['Order ID', 'Product'])
print(multi_indexing)

# Use 'inplce = True' to modify the DataFrame in place.

shopping_df.set_index('Order ID', inplace=True)
print(shopping_df)

# Setting the index while reading the 'csv' file.

shopping_df = pd.read_csv('shopping_data_v2.csv', index_col='Order ID')
print(shopping_df)

""" 
pd.DataFrame.reset_index:-
--> pd.DataFrame.reset_index(drop=False, inplace=False)
    1. Returns a DataFrame with the 'new index' or 'None' if 'inplace=True'.
    2. Returns the index of the DataFrame, and use the default one instead.
"""
print(shopping_df.reset_index())

# We can use the 'drop' parameter to avoid the old index being added as a column:

print(shopping_df.reset_index(drop=True))

print(shopping_df)

shopping_df.reset_index(inplace=True)
print(shopping_df)

# Accessing data with updated Index: 'loc' 

shopping_df.set_index("Order ID", inplace=True)
print(shopping_df)


# The following code throws error.

print(shopping_df.loc[2,'Product':'Order Date'])

# Changing the index changes the way we access the data with 'loc' .

shopping_df.set_index("Order ID", inplace=True)

# Raises an error.

print(shopping_df.loc[2,'Product':'Order Date'])

print(shopping_df.loc[176560,'Product':'Order Date'])

""" 
Using methods based on Index:- 

pd.DataFrame.sort_index:-
--> pd.DataFrame.sort_index(axis=0, ascending=True, inplace=False, key=None)
    1. Returns a new DataFrame sorted by label if 'inplace=False', otherwise updates the original DataFrame and returns None.
"""

print(shopping_df.sort_index())

""" 
--> By default, it sorts in ascending order, to sort in descending order, use 'ascending=False'.
"""

print(shopping_df.sort_index(ascending=False))

""" 
--> A key function can be specified which is applied to the index before sorting.
"""

sort_df = pd.DataFrame({"int_values":[1,2,3,4]},index=['a','E','F','g'])
sorted_df = sort_df.sort_index(key=lambda x:x.str.lower())
print(sorted_df)

""" 
pd.DataFrame.first_valid_index:-
--> Returns the 'index' of the first non-NA/null value.
"""

import numpy as np

df1 = pd.DataFrame({
    "name":[np.nan,"Lokesh","Sai Kishore"],
    "place":[np.nan,"Bangalore","Hyderabad"]
})
print(df1)
print(df1.first_valid_index())

""" 
Filtering:-

Creating Masks:-
"""

print(shopping_df['Price Each'])

price_filter = shopping_df["Price Each"] > 100
print(price_filter)

""" 
--> The boolean expression returns a series containing 'True' and 'False' boolean values.
--> You can use this series to select a subset of rows from the original dataframe, corresponding to the 'True' values in the series.
"""

print(shopping_df[price_filter])

print(shopping_df.loc[price_filter,'Price Each'])

print(shopping_df.loc[price_filter,'Price Each':'Purchase Address'])

# Using Logical and Relational Operators:-

price_range_filter = (shopping_df['Price Each'] > 10) & (shopping_df['Price Each'] < 20)
print(shopping_df.loc[price_range_filter])

# Excluding the rows which satisfy the given condition.

neg_filter = (shopping_df['Price Each'] >= 10)
print(shopping_df[neg_filter])
print(shopping_df[~neg_filter])

# Using 'str' methods:-

print(shopping_df['Product'])

filter = shopping_df['Product'].str.contains('Headphones')
print(shopping_df[filter])

filter = shopping_df['Product'].str.endswith('Cable')
print(shopping_df[filter])

""" 
pd.DataFrame.query:-
--> pd.DataFrame.query(expression, inplace=False, **kwargs)
    1. Returns a DataFrame resulting from the provided query expression.
"""

df = pd.DataFrame({
    'A':[1,6],
    'B':[10,3],
    'C':[10,5]
})
condition = df.query('B == C & A < B')
print(condition)

# The above expression is also equivalent to:

print(df[(df.B == df.C) & (df.A < df.B)])