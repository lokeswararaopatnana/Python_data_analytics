import pandas as pd

# Q:- Load the dataset into a dataframe using 'read_csv'.?

film_df = pd.read_csv('film.csv')

# Q:- Get the shape of the dataset.?

shape_film_df = film_df.shape
print(shape_film_df)

# Q:- Get the names of the first 6 films.?
film_names = film_df.loc[0:5,'Title']
print(film_names)

# Q:- Generate descriptive statistics for the dataset, including all datatypes.?
film_describes = film_df.describe(include="all")
print(film_describes)

# Q:- Change the 'Popularity' at the third index to 70.?
film_df.at[3,'Popularity'] = 70
print(film_df.at[3,'Popularity'])

# Q:- Get the data in the first two columns for the last five rows in the dataset.?

print(film_df.loc[1123:1127,'Year':'Length'])