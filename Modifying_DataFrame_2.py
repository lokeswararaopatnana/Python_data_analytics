import pandas as pd
import numpy as np

# Loading the shopping dataset.
shopping_df = pd.read_csv('shopping_data_v2.csv')

# Loading the film dataset.
film_df = pd.read_csv('film.csv')

# sample dataset.
people = {
    "first":["Virat","Rohit","Shubman"],
    "last":["Kohli","Sharma","Gill"],
    "eamil":["Virat.Kohli@gmail.com","Rohit.Sharma@gmail.com","Shubman.Gill@gmail.com"]
}
people_df = pd.DataFrame(people)

""" 
Add/Delete Columns:-
--> We can create new column by assigning a 'Series' as showen below.
"""
print(shopping_df)

shopping_df['Store'] = pd.Series(['Amazon','Flipkart','Walmart'])
print(shopping_df)

# Creating a new 'Series' and adding it to a 'DataFrame'.

print(people_df)
people_df["full_name"] = (people_df['first'] + ' ' + people_df['last'])
print(people_df)

# Splitting columns:-

people = {
    "full_name": ["Virat Kohli","Rohit Sharma","Shreyas Iyer"],
    "email": ["ViratKohli@gmail.com","RohitSharma@gmail","ShreyasIyer@gmail.com"]
}
people_df = pd.DataFrame(people)
print(people_df)

splitting_full_name = people_df["full_name"].str.split(' ')
splitting_full_name = people_df["full_name"].str.split(' ',expand=True)
splitting_full_name[['first_name','last_name']] = people_df["full_name"].str.split(' ',expand=True)
splitting_full_name = splitting_full_name[['first_name','last_name']]
print(splitting_full_name)

""" 
pd.concat:-
--> pd.concat(objs, axis=0, ignore_index=False, keys=None)
    1. Concatenates pandas objects along a particular axis.
    2. 'objs' is sequence or mapping of Series or DataFrame objects.
    3. The 'axis' to concatenate along.
    4. If 'ignore_index' is True, then it does not use the index values along the concatenation axis. The resulting axis will be labeled 0,..,n-1.
"""
people = {
    "first":["Virat","Rohit","Shubman"],
    "last":["Kohli","Sharma","Gill"],
    "eamil":["Virat.Kohli@gmail.com","Rohit.Sharma@gmail.com","Shubman.Gill@gmail.com"]
}
people_df = pd.DataFrame(people)
print(people_df)

ages_and_hobbies = {
    "age":[23,24,23,26],
    "hobbies":["Cooking","Cricke","FootBall","Painting"]
}
ages_and_hobbies_df = pd.DataFrame(ages_and_hobbies)
print(ages_and_hobbies_df)

concatenation_df = pd.concat([people_df,ages_and_hobbies_df], axis="columns")
print(concatenation_df)

# We can also concatenate 2 'series' objects.

name = np.array(["Lokesh","Mellisa"])
gender = np.array(["Male","Female"])

name_series = pd.Series(name)
gender_series = pd.Series(gender)

user_df = pd.concat([name_series,gender_series],axis=1)
print(user_df)

# You can also create column names using the 'keys' option.

user_df = pd.concat([name_series,gender_series],axis=1,keys=["name", "gender"])
print(user_df)

""" 
pd.DataFrame.drop:-
--> pd.DataFrame.drop(labels=None, axis=0, index=None, columns=None,inplace=False)
    1. 'labels' os the index or column labels to drop.
    2. 'axis' specifies the axis to drop the labels from.
    3. 'index' is an alternative to specifying the axis (labels, axis=0 is equivalent to index=labels).
    4. 'columns' is an alternative to specifying the axis(labels, axis=1 is equivalent to columns=labels).
"""

print(shopping_df)

shopping_df.drop(columns=['Store'], inplace=True)
print(shopping_df)

# Alternate way of deleting columns.
alternate_dropping = shopping_df.drop(labels='Order Date',axis="columns")
print(alternate_dropping)

""" 
Add/Delete Rows:-
pd.DataFrame.append:-
--> pd.DataFrame.append(other, ignore_index=False,sort=False)
    1. Columns in 'other' that are not in the dataframe are added as new columns.
    2. If 'ignore_index' is True, the resulting axis will be labeled 0, 1,..., n-1.
    3. If 'sort' is True, then it will sort the columns.
"""

people = {
    "first": ["Surya","Shreyas","KL","Ravindra"],
    "last":["Kumar","Iyer","Rahul","Jadeja"],
    "email":["suryakumar@gmail.com","shreyas@gmail.com","klrahul@gmail.com","jadeja@gmail.com"]
}
people_df = pd.DataFrame(people)
print(people_df)

# The following line of code throws an error, because we didn't assign an index to the new row.

# updating_people_df = people_df.append({'first':'Virat','last':'Kohli','eami':'virat@gmail.com'})
# print(updating_people_df)

# If 'ignore_index' is set to 'True', the row will be indexed automatically. 

updating_people_df = people_df.append({'first':'Virat','last':'Kohli','email':'virat@gmail.com'}, ignore_index=True)
print(updating_people_df)

# We can append a dataframe too.

people = {
    "first":["Jasprit","Mohammadh"],
    "last" : ["Bhumrah","Shami"],
    "email" : ["jasprit@gmail.com","shami@gmail.com"]
}
people_df_2 = pd.DataFrame(people)
print(people_df_2)

updating_two_dfs = people_df.append(people_df_2, ignore_index=True)
print(updating_two_dfs)
updating_two_dfs_ = people_df.append(people_df_2, ignore_index=True,sort=True)
print(updating_two_dfs_)

# pd.DataFrame.drop:-

print(shopping_df)

dropping_rows = shopping_df.drop(index=[1,2])
dropping_rows = shopping_df.drop(labels=[4,3], axis="index")
# print(dropping_rows)
dropping_rows_product = shopping_df['Product'].drop(labels=[1,2])
print(dropping_rows_product)

""" 
Merge DataFrames:-
--> pd.DataFrame.merge(right, on=None)
    1. Returns a DataFrame of the two merged objects.
    2. 'right'is the object to merge with.
    3. 'on': the columns to join on. It must be in both the dataframe objects.
    (Similar to the SQL JOIN operation)
"""
people = {
    "full_name":["Mohamed Siraj","Kuldeep Yadav"],
    "email":["siraj@gmail.com","yadav@gmail.com"],
    "place":["Hyderabad","Lucknow"]
}
df1 = pd.DataFrame(people)
print(df1)

location = {
    "place":["Hyderabad","Lucknow"],
    "state":["Telangana","Uttar Pradesh"]
}
df2 = pd.DataFrame(location)
print(df2)

merging_two_dfs = df1.merge(df2, on="place")
print(merging_two_dfs)

""" 
Sorting:-

pd.DataFrame.sort_values:-
--> pd.DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, ignore_index=False)
    1. Sort the values along either axis.
    2. 'by' is the name or list of names to sort by.
    3. 'axis' is the axis to be sorted.
    4. If 'ignore_index' is True, then the resulting axis will be labeled 0,1,...,n-1.
"""
print(film_df)

sorting_values_by_year = film_df.sort_values(by='Year')
print(sorting_values_by_year)

sorting_values_by_popularity = film_df.sort_values(by='Popularity', ascending=False)
print(sorting_values_by_popularity)

sorting_values_by_popularity_index = film_df.sort_values(by='Popularity', ascending=False,ignore_index=True)
print(sorting_values_by_popularity_index)


sorting_values_by_popularity_year_index = film_df.sort_values(by=['Year','Popularity'], ascending=[True,False])
print(sorting_values_by_popularity_year_index)

# We can use 'sort_values' with Series too.

print(film_df['Length'].sort_values(ascending=False))

""" 
NLargest and NSmallest:-
--> 'nlargest' returns the first 'n' rows ordered by columns in descending order.
"""
print(film_df['Length'])
print(film_df["Length"].nlargest(10))
print(film_df.nlargest(15,['Popularity','Length']))

""" 
--> 'nsmallest' returns the first 'n' rows ordered by columns in ascending order.
"""

print(film_df['Year'].nsmallest(5))
print(film_df.nsmallest(5,'Length'))