import pandas as pd 

# using covid dataset.

# Q:- Load the dataset into a dataframe using 'read_csv'.?
covid_df = pd.read_csv('italy-covid-daywise.csv')
print(covid_df)

# Q:- Make the 'date' column as the index.?
covid_df.set_index('date', inplace=True)
print(covid_df)

# Q:- Sort the records in the dataset based on the 'date' column.?
print(covid_df.sort_index())

# Q:- Find out on which days more than 1000 cases were reported.?
cases_filter = covid_df['new_cases'] > 1000
print(covid_df[cases_filter])

# Q:- Fetch the 'new_cases' and 'new_tests' reported in the month of June.?

filter = covid_df['date'].str.contains('-06-')
june_report = covid_df[filter]
print(june_report['new_cases'])
print(june_report['new_tests'])

# Q:- Find the first row in which valid 'new_tests' were reported.?

check_first_valid = covid_df["new_tests"]
print(check_first_valid.first_valid_index())

# -----------------------------------------------------------

# using film dataset
# Loading the film dataset

film_df = pd.read_csv('film.csv')
print(film_df)

# Q:- Make the year and title as indices.?

film_df.set_index(['Year', 'Title'],inplace=True)
print(film_df)

# Q:- Fetch the films where the 'actor' name contains 'Jasen' and the 'popularity' is greater than 50.?

popularity_greater_than_50 = (film_df['Popularity'] > 50)
filter_1 = film_df[popularity_greater_than_50]
accessing_actor_name = (film_df['Actor'].str.contains('Jasen')) 
print(film_df[accessing_actor_name])
