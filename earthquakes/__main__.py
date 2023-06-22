
"""
import pandas as pd

# Define the file path
file_path = "data/earthquakes1900_2023.csv"

# Read the CSV file into a DataFrame
eq = pd.read_csv(file_path, delimiter=',', parse_dates=['Date'], index_col="Date")

eq.sort_values(by='Date', ascending=False, inplace=True)

#df = pd.read_csv("data/earthquakes1900_2023.csv", delimiter=',', parse_dates=['date'])
#df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
#df["Depth"] = df["Depth"].replace(np.nan, 0)
#df = df[['Latitude', 'Longitude', 'Depth', 'Magnitude', 'nst']]


df = pd.read_csv("Earthquake_db.csv",  delimiter=',', parse_dates=['Date'])
df = df.set_index('Date')

"""

from import_csv import import_csv
from plots import explore_plot
from features import *
from preprocessing import *


#Import csv file
path = 'earthquakes/data/earthquakes_turkey.csv'
df = import_csv(path)


# <====================== EXPLORATION ======================>
#explore_plot(df)


"""
# <====================== FEATURE ENGINEERING ======================>
#Split Date Columns
df = add_split_date(df, split = 'year')
df = add_split_date(df, split = 'month')
df = add_split_date(df, split = 'day')

#Time between events
df = add_inter_event_duration(df)

#Add rolling mean to mag column 
df = add_rolling_statistic(df, column='magnitude', window_size=5, fill_value="mean", statistic='mean')

#Add rolling avg to mag column 
df = add_rolling_statistic(df, column='magnitude', window_size=5, fill_value="mean", statistic='avg')

#Add rolling std to mag column 
df = add_rolling_statistic(df, column='magnitude', window_size=5, fill_value="mean", statistic='std')

#Add rolling max to mag column 
df = add_rolling_statistic(df, column='magnitude', window_size=5, fill_value="mean", statistic='max')

#Add rolling min to mag column 
df = add_rolling_statistic(df, column='magnitude', window_size=5, fill_value="mean", statistic='min')

#Drop remaining Na
df = drop_na(df)

#Set date as index column
df = set_index(df, col_name='date')

#Drop columns
#df = drop_col()

#Add cluster for Lat and Long
df = add_kmeans(df, num_clusters=5)

#df = df.head(100)

#df = add_mean_mag_location(df, 1)
"""
print(df.head(5))
# <====================== PREPROCESSING ======================>

#Splitting training and test set
train, test = split_data(df, 0.75)


#Normalizing training and test set
train = normalize_columns(df, ['depth', 'magnitude'])
test= normalize_columns(df, ['depth', 'magnitude'])

print(df.head(5))
