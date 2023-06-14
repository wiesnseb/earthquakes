import pandas as pd

# Define the file path
file_path = "data/earthquakes1900_2023.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

print(df.head(10))