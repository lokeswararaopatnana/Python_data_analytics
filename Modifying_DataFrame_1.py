import pandas as pd 
import numpy as np

# Loading shopping dataset
shopping_df = pd.read_csv('shopping_data_v2.csv')
print(shopping_df)

# Modifying Column Names:-

people = {
    "First Name" : ["Lokesh","Sai Kishore","Hari Kirshna"],
    "Last Name" : ["Patnana","Reddi","Allu"],
    "Email ID" : ["lokeshpatnana@gmail.com","saikishorereddi@gmail.com","harikrishnaallu@gmail.com"]
}
people_df = pd.DataFrame(people)
print(people_df)
print(people_df.columns)

people_df.columns = ["F Name","L Name","Email"]
print(people_df)

people_df.columns = people_df.columns.str.replace(" ", "_")
print(people_df)

people_df.columns = [x.lower() for x in people_df]
print(people_df)

""" 
pd.DataFrame.rename:-
--> pd.DataFrame.rename(mapper=None, index=None, columns=None, axis=None, inplace=False)
    1. 'mapper' is the dict-like or functions transformations to apply to that axis values.
    2. 'axis' is the axis to target with mapper.
    3. 'index' is an alternative to specifying 'axis' (mapper, axis=0 is equivalent to index=mapper).
    4. 'columns' is an alternative to specifying 'axis' (mapper, axis=1 is equivalent to column=mapper).
"""
print(shopping_df)

changing_col_names = shopping_df.rename(columns={'Price Each':'Price','Product':'Product Name'})
print(changing_col_names)

# Alternate way of renaming columns.

changing_col_names = shopping_df.rename(mapper={'Price Each':'Price','Product':'Product Name'}, axis=1)
print(changing_col_names)

""" 
After carefully checking the changes, use 'inplace=True' to make the changes reflect in the actual dataframe.
"""

shopping_df.rename(columns={'Price Each':'Price','Product':'Product Name'}, inplace=True)
print(shopping_df)

""" 
Modifying Rows:-
--> Trying to update a value as shown below will raise an error. Instead, values should be updating using 'loc' or 'iloc'.
"""
# Here this way of updating is not supported.
filt = (people_df['email'] == 'lokeshpatnana@gmail.com')
people_df[filt]['l_name'] = "P"
print(people_df)

# Updating values with 'loc':-

people_df.loc[1] = ['Arun kumar','Marrapu','arun.marrapu@gmail.com']
print(people_df)

people_df.loc[0,['f_name','email']] = ['Lokeswara Rao','lokeswararaopatnana@gmail.com']
print(people_df)

people_df.loc[[1,2],['f_name']] = "Apple"
print(people_df)

""" 
Methods to Update Data:-

pd.Series.apply:-
--> pd.Series.apply(func, convert_dtype=True, args=(), **kwds)
    1. Applies the function on the values in the Series.
    2. 'func' is the Python function or NumPy ufunc to apply.
    3. 'args' are the positional arguments passed to func after the series value.
    4. '**kwds' are the additional keyword arguments passed to func.
"""

def change_product_name(product_name):
    new_name = product_name.lower()
    new_name = new_name.replace(" ", "_")
    return new_name

changing_product_name = shopping_df["Product Name"].apply(change_product_name)
print(changing_product_name)

# Can also use lambda functions.

using_lambda = shopping_df["Price"].apply(lambda x: x + 5)
print(using_lambda)

length_product = shopping_df["Product Name"].apply(len)
print(length_product)

""" 
--> Apply can be used directly with dataframes too. But, the Objects passed to the function are 'Series' objects.
"""

print(shopping_df.apply(len))

""" 
pd.DataFrame.applymap:-
--> pd.DataFrame.applymap(fun)
    1. Applies the function to every element of the DataFrame.
    2. 'func' is the Python function to apply.
"""

people = {
    "full_name":["Lokesh Patnana","Sai Reddi","Hari Allu"],
    "email":["lokesh.patnana@gmail.com","sai.reddi@gmail.com","hari.allu@gmail.com"]
}
people_df = pd.DataFrame(people)
print(people_df)
print(people_df.applymap(len))
print(people_df.applymap(str.lower))

""" 
pd.Series.map:-
--> pd.Series.map(arg)
    1. It is used to map values of the Series according to input correspondence.
    2. 'arg' is used for substituting each value in a Series with another value. It may be a function, a dict or a Series.

--> Values in the Series that are NOT in the dictionary are converted to NaN.
"""
print(shopping_df)

using_map = shopping_df["Product Name"].map({'Google Phone':'Google Pixel','iPhone':'iPhone 6'})
print(using_map)

""" 
pd.Series.replace:-
--> pd.Series.replace(to_replace=None, value=None, inplace=False)
    1. Applies the function on the values in the Series.
    2. 'to_replace' is/are the value(s) to replace. It may be a str, list, dict, Series, int etc.
    3. The 'value' to replace any values matching 'to_replace' with.
"""

using_replace = shopping_df["Product Name"].replace({'Google Phone':'Google Pixel','iPhone':'iPhone 6'})
print(using_replace)

# Replace can be used with dataframes too.

using_replace_to_replace = shopping_df.replace(to_replace=["USB-C Charging Cable", "Lightining Charging Cable"], value="Charging Cable")
print(using_replace_to_replace)