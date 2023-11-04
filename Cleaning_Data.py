import pandas as pd
import numpy as np 

# Loading shopping dataset
shopping_df = pd.read_csv('shopping_data.csv')

# Loading stackverflow dataset
stackoverflow_df = pd.read_csv('survey_results_public.csv')

# Loading film dataset
film_df = pd.read_csv('film.csv')

""" 
Cleaning Data:-

Handling Missing Values:-

pd.DataFrame.isna:-
--> pd.DataFrame.isna()
    1. Return a boolean same-sized object indicating if the values are 'NA'.
    2. 'None' or 'numpy.NaN' are considered as 'NA' values.
    3. Characters such as '' or 'numpy.inf' are not considered as 'NA' values.
"""
print(shopping_df)
print(shopping_df.isna())
print(shopping_df['Order ID'].isna())
print(shopping_df['Order ID'].isna().sum())

# To find the number of 'NaN' objects within the given series.
print(shopping_df.isna().sum())

""" 
pd.DataFrame.fillna:-
--> pd.DataFrame.fillna(value=None, limit=None)
    1. Fill NA/NaN values with the given value.
    2. 'value' is used to fill NaN. It can be a single value(e.g. 0) or a dict, Series, DataFrame wtc. It cannot be a list.
    3. 'limit' is the maximum number of entries along the entire 'axis' where NaNs will be filled.
"""

print(shopping_df.fillna(0))
print(shopping_df.fillna('MISSING'))

# You can fill appropriate missing values for each of the columns.

values = {'Order ID':'00','Product':'xxx',"Quantity Ordered":"1","Price Each":"100","Order Date":"04/07/19 23:30","Purchase Address":"Jp Nagar, Miyapur, Telangana, 500049"}
print(shopping_df.fillna(value=values).head(10))
print(shopping_df.fillna(value=values, limit=2).head(10))

""" 
pd.DataFrame.dropna:-
--> pd.DataFrame.dropna(axis=0, how='any', subset=None, inplace=False)
    1. Removes missing values.
    2. Returns a DataFrame with the 'NA' entries dropped from it.
"""
print(shopping_df.dropna())

# Drop the rows where all the elements are missing.
print(shopping_df.dropna(how='all'))

# Drop the columns where atleast one element is missing.
print(shopping_df.dropna(axis='columns',how='any'))

# Can also defined the specific columns to look for the missing values using the 'subset' parameter.
print(shopping_df.dropna(subset=['Order ID','Product']))

# More Examples:-
people = {
    'first':['Lokesh','Glenn','Mitchell','Emma',np.nan,None,'NA'],
    'last':['Patnana','Philips','Santener','Collins',np.nan,np.nan,'Missing'],
    'email':['lokeshpatnana@gmail.com','glennphillips@gmail.com','mitchellsantener@gmail.com',None,None,np.nan,'Asynchronous@gmail.com'],
    'age':['23','44','45','46',None,None,'Missing']
}
people_df = pd.DataFrame(people)
print(people_df)
print(people_df.isna())

# If both the 'email' and 'last' is NA, then drops the row.
print(people_df.dropna(axis='index',how='all',subset=['last','email']))

# If either the 'email' or 'last' is NA, then drops the row.
print(people_df.dropna(axis='index',how='any',subset=['last','email']))

# Using 'replace' to handle other missing values.
people = {
    'first':['Lokesh','Missing','Mitchell','Emma',np.nan,None],
    'last':['Patnana','Philips','Santener','Collins',np.nan,np.nan,],
    'email':['NA','glennphillips@gmail.com','mitchellsantener@gmail.com',None,np.nan,'Asynchronous@gmail.com'],
    'age':['23','44','45','46',None,None,]
}
people_df = pd.DataFrame(people)
print(people_df)
people_df.replace('NA',np.nan,inplace=True)
people_df.replace('Missing',np.nan,inplace=True)
print(people_df)
print(people_df.isna())

# We can also specify additional strings to be recognized as 'NA/NaN' while loading the data itself.
na_vals = ['NA','Missing']
survey_df = pd.read_csv('survey_results_public.csv', index_col='Respondent', na_values=na_vals)
print(survey_df)

""" 
Handling Duplicates:-

pd.DataFrame.duplicated:-
--> pd.DataFrame.duplicated(subset=None, keep='first')
    1. It returns a boolean series for each of the duplicated rows.
    2. 'subset' is used to only consider certain columns for identifying duplicates.
    3. 'keep' determines which duplicates (if any) to mark.
        1. 'first' marks all the duplicates as True except for the first occurrence.
        2. 'last' marks all the duplicates as True except for the last occurrence.
        3. 'False' marks all the duplicates as True.
"""
print(shopping_df.loc[25:35])
print(shopping_df.loc[25:35].duplicated())
print(shopping_df.loc[25:35].duplicated(keep='last'))
print(shopping_df.loc[25:35].duplicated(subset=['Product']))
print(shopping_df.loc[25:35].duplicated(subset=['Product'],keep='last'))

""" 
pd.DataFrame.drop_duplicates:-
--> pd.DataFrame.drop_duplicates(subset=None, keep='first', inplace=False)
    1. It returns a dataframe with the duplicated rows removed.
    2. 'subset' is used to only consider certain columns for identifying duplicates.
    3. 'keep' determines which duplicates (if any) to mark.
        1. 'first' marks all the duplicates as True except for the first occurrence.
        2. 'last' marks all the duplicates as True except for the last occurrence.
        3. 'False' marks all the duplicates as True.
"""
shopping_sample = shopping_df[25:35]
print(shopping_sample.drop_duplicates())
print(shopping_sample.drop_duplicates(keep='last'))
print(shopping_sample.drop_duplicates(subset=['Product']))

# Changing Datatypes:-

# Using 'astype' :-
print(shopping_df.dtypes)
print(shopping_df)
print(shopping_df['Price Each'].astype('float32'))

people = {
    'first':['Lokesh','Missing','Mitchell','Emma',np.nan,None],
    'last':['Patnana','Philips','Santener','Collins',np.nan,np.nan,],
    'email':['NA','glennphillips@gmail.com','mitchellsantener@gmail.com',None,np.nan,'Asynchronous@gmail.com'],
    'age':['23','44','45','46',None,None,]
}
people_df = pd.DataFrame(people)
print(people_df)

# The following code raises an error, because 'age' is string.

# print(people_df['age'].mean())

# If we try to convert it to 'int' it still raises an error, because there are 'NaN' values in the 'age' column.
# people_df['age'] = people_df['age'].astype(int)
# print(people_df['age'])

# Internally, 'NaN' values are represented as 'float'. So, we can convert the age column into float instead.

people_df['age'] = people_df['age'].astype(float)
print(people_df['age'])
print(people_df['age'].mean())

# The following code raises an error, because 'YearsCode' contains string values.
print(survey_df)
# survey_df['YearsCode'] = survey_df['YearsCode'].astype('float')
print(survey_df['YearsCode'].unique())
survey_df['YearsCode'].replace('Less than 1 year',0,inplace=True)
survey_df['YearsCode'].replace('More than 50 years', 55,inplace=True)
survey_df['YearsCode'] = survey_df['YearsCode'].astype('float')
print(survey_df['YearsCode'].median())

# Note: We can convert everything in the dataframe to a single datatype at once, using 'dataframe.astype(dtype)'.

# Converting into datetime objects:-
print(shopping_df.dtypes)

""" 
    1. The data type of 'Order Date' is currently 'object'.
    2. It can be converted into a 'datetime' column using the 'pd.to_datetime' method.
"""
shopping_df['Order Date'] = pd.to_datetime(shopping_df['Order Date'])
print(shopping_df.dtypes)

# Adding new columns related to Datetime.
data = shopping_df.copy()
print(data)

data['Hour'] = data['Order Date'].dt.hour
data['Minute'] = data['Order Date'].dt.minute
data['Second'] = data['Order Date'].dt.second
data['Day'] = data['Order Date'].dt.day
data['Month'] = data['Order Date'].dt.month
data['Year'] = data['Order Date'].dt.year
print(data.head())
print(data['Order Date'].dt.day_name())