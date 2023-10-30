import pandas as pd
"""
Pandas - Brief Intro:-
1. Data Analysis library which makes it easy to read and work with different types of data.
    --> Load large amounts of data into python.
    --> Work with different file formats (csv,sql,excel,etc.)
    --> Work with time - series data (stock prices,sales data, etc.)
2. Cleanind the Data and Handiling Missing Values.
3. Built on top of NumPy and has great performance.
"""
# Reading & Writing files:-
    # Reading/Loading the data from a csv file.

shopping_df = pd.read_csv('shopping_data_v2.csv')

""" 
Dataframe:-
--> A dataframe is like a dictionary of lists, but with much more functionality.
    1. A table of data (Rows and Columns).
    2. 2-Dimesional Data Structure.
"""

# Properties of DataFrame:-
# Shape of a dataframe:

shopping_df_shape = shopping_df.shape
print(shopping_df_shape)

# 'df.columns' returns the columns of a DataFrame.

shopping_df_columns = shopping_df.columns
print(shopping_df_columns)

# 'df.dtypes' returns the datatypes of each columns in a DataFrame.

shopping_df_dtypes = shopping_df.dtypes
print(shopping_df_dtypes)

""" 
df.head:-
--> df.head(n=5)
    1. Returns the first 'n' rows.
    2. For negative values of 'n', this function returns all the rows except for the last 'n' rows.
"""

shopping_df_head = shopping_df.head()
shopping_df_head = shopping_df.head(10)
shopping_df_head = shopping_df.head(-1000)
print(shopping_df_head)

""" 
'df.tail:-
--> dg.tail(n=5)
    1. Returns the last 'n' rows.
    2. For negtive values of 'n', this function returns all rows except for the first 'n' rows.
"""

shopping_df_tail = shopping_df.tail()
shopping_df_tail = shopping_df.tail(15)
shopping_df_tail = shopping_df.tail(-100)
print(shopping_df_tail)

""" 
df.describe:-
--> df.describe(include =None, exclude =None)
    1. Generates descriptive statistics.
    2. Analyzes both numeric and object series as well as mixed data types.
    3. 'include' : A list of data types to include in the result.
    4. 'exclude' : A list of data types to omit in the result.
"""

shopping_df_describe = shopping_df.describe()
shopping_df_describe = shopping_df.describe(include = ['object'])
shopping_df_describe = shopping_df.describe(include="all")
shopping_df_describe = shopping_df.describe(exclude=['object'])
print(shopping_df_describe)

""" 
df.info:-
--> df.info()
    Print a concise summary of a DataFrame including the dtypes of the columns, memory usage, etc.
"""

shopping_df_info = shopping_df.info()
print(shopping_df_info)

# Comparision with Dictionary of Lists:-

people = {
    "first_name":["Patnana","Reddi","Allu"],
    "last_name":["Lokesh","Sai Kishore","Hari Krishna"],
    "email":["lokeshpatnana1@gmail.com","saikishorereddi9640@gmail.com","hariallu946@gmail"]
}

# Creating DataFrame from a dictionary of lists.
df = pd.DataFrame(people)
print(df)
print(people["first_name"])
print(df["last_name"])
print(type(df["first_name"]))

""" 
Series:-
    A Series is like a list of data, but with much more functionality.
"""
# Creating a Series from a list of Data.
series_df = pd.Series([1,2,3,4,"abcd"])
print(series_df)

print(type(shopping_df["Product"]))

""" 
Accessing Data:-

Indexing:-
    We can retrieve a specific value from a series using the indexing notation [].
"""
print(shopping_df["Product"][200])

""" 
df.iloc:-
--> Accessing by position numbers.
--> df.iloc()
    1. Purely integer - location based indexing for selection by postion.
    2. Unlike in 'loc', if a slice object with indices is passed then stop is excluded.
"""

shopping_df_iloc = shopping_df.iloc[[0,2],[0,3]]
shopping_df_iloc = shopping_df.iloc[0,1]
print(shopping_df_iloc)

""" 
df.loc:-
--> Accessing by label/name.
--> df.loc()
    1. Access a group of rows and columns by label(s) or a boolean array.
    2. Allowed inputs are:
            # single label
            # list or array of labels
            # slice object with labels (both start and stop are included)
            # boolean array of the same length as the axis being sliced.
"""
shopping_df_loc = shopping_df.loc[980]
shopping_df_loc = shopping_df.loc[[980,456]]
shopping_df_loc = shopping_df.loc[[980,456],['Product','Order Date']]
shopping_df_loc = shopping_df.loc[45,'Product']
print(shopping_df_loc)
print(shopping_df[0:3])

""" 
df.at:-
--> df.at()
    1. Access a single value for a row/column label pair.
    2. Can also set/update a value at specified row/column pair.
    3. Use 'at' only if you need to get or set a single value in a DataFrame or Series.
""" 
shopping_df_at = shopping_df.at[50000,'Product']
print(shopping_df_at)

df_copy = shopping_df.copy()
print(df_copy.at[50000,'Product'])
df_copy.at[50000,'Product'] = "Micro USB Cable"
print(df_copy.at[50000,'Product'])

# Slicing:-
print(shopping_df[0:3])

# When slicing using 'loc' the start and stop indices are included.

print(shopping_df.loc[7:9])
print(shopping_df.loc[23:45,'Product'])
print(shopping_df.loc[10:20,'Product':'Order Date'])

# When slicing using 'iloc' the stop index is excluded.

print(shopping_df.iloc[1:2,0:3])
print(shopping_df.iloc[1,0:3])