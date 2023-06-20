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