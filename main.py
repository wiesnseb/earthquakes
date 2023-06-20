import pandas as pd

# Define the file path
file_path = "data/earthquakes1900_2023.csv"

# Read the CSV file into a DataFrame
eq = pd.read_csv(file_path, delimiter=',', parse_dates=['Time'], index_col="Time")

eq.sort_values(by='Time', ascending=False, inplace=True)

#print(eq.head())

print(len(eq.index))



df = pd.read_csv("Earthquake_db.csv",  delimiter=',', parse_dates=['Date'], index_col="Date")
print(len(df.index))