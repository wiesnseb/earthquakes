import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm



def add_split_date(df, split='year', date_column='date'):
    df[split] = getattr(pd.to_datetime(df[date_column]).dt, split)
    return df


def add_inter_event_duration(df, date_column='date'):
    df["inter_event_duration"] = df[date_column].diff().apply(lambda x: x.total_seconds())
    return df


def rolling_n (df, window=10):
    desc = "mag_roll_" + window
    df[desc] = df["magnitude"].rolling(window=10).mean()


def add_rolling_mean(df, col_name, window_size=10):
    rolling_mean_col = f"rolling_{window_size}_{col_name}"
    df[rolling_mean_col] = df[col_name].rolling(window_size).mean()
    return df


def add_rolling_statistic(df, col_name, window_size=10, statistic='mean', fill_value=None):
    valid_statistics = ["mean", "std", "avg", "max", "min"]

    if statistic not in valid_statistics:
        raise ValueError("Invalid statistic. Valid options are: mean, std, avg, max, min")

    rolling_statistic_col = f"rolling_{window_size}_{col_name}_{statistic}"
    
    if statistic == "mean":
        df[rolling_statistic_col] = df[col_name].rolling(window_size).mean()
    elif statistic == "std":
        df[rolling_statistic_col] = df[col_name].rolling(window_size).std()
    elif statistic == "avg":
        df[rolling_statistic_col] = df[col_name].rolling(window_size).sum() / window_size
    elif statistic == "max":
        df[rolling_statistic_col] = df[col_name].rolling(window_size).max()
    elif statistic == "min":
        df[rolling_statistic_col] = df[col_name].rolling(window_size).min()

    if fill_value is not None:
        if fill_value == "mean":
            fill_value = df[col_name].mean()
        df[rolling_statistic_col].fillna(fill_value, inplace=True)
    else:
        df[rolling_statistic_col].fillna(np.nan, inplace=True)

    return df

def drop_na(df, inplace=True):
    df = df.dropna(inplace = inplace)
    return df


def set_index(df, col_name):
    df = df.set_index(col_name)
    return df


def drop_col(df, col_name, axis=1):
    df = df.drop(col_name, axis=axis)
    return df


def add_kmeans(df, num_clusters, lat_col='latitude', long_col='longitude'):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(df[[lat_col, long_col]])
    df['cluster_id'] = kmeans.labels_
    
    return df


from tqdm import tqdm

def add_mean_mag_location(df, radius, lat_col='latitude', long_col='longitude', mag_col='magnitude'):
    """
    Calculates the mean magnitude of earthquakes within a certain radius for every row in the DataFrame.
    
    Arguments:
    df -- DataFrame containing earthquake data
    lat_col -- Column name for latitude values
    long_col -- Column name for longitude values
    mag_col -- Column name for magnitude values
    radius -- Radius in kilometers
    
    Returns:
    df -- DataFrame with an additional column 'Mean Magnitude' containing the mean magnitude of earthquakes
          within the specified radius for each row in the DataFrame
    """
    
    # Convert radius from kilometers to degrees (approximation)
    radius_deg = radius / 111.12
    
    # Calculate mean magnitude for each row in the DataFrame
    mean_magnitudes = []
    total_rows = len(df)
    
    # Use tqdm to create a progress bar
    with tqdm(total=total_rows, desc="Calculating mean magnitudes") as pbar:
        for index, row in df.iterrows():
            lat = row[lat_col]
            long = row[long_col]
            magnitude = row[mag_col]

            # Convert coordinates to radians
            lat_rad = radians(lat)
            long_rad = radians(long)

            # Calculate the mean magnitude for earthquakes within the specified radius
            magnitudes_within_radius = []
            for _, row_inner in df.iterrows():
                lat_inner = row_inner[lat_col]
                long_inner = row_inner[long_col]
                magnitude_inner = row_inner[mag_col]

                # Convert coordinates to radians
                lat_inner_rad = radians(lat_inner)
                long_inner_rad = radians(long_inner)

                # Calculate the distance between two earthquakes using the Haversine formula
                dlat = lat_inner_rad - lat_rad
                dlong = long_inner_rad - long_rad
                a = sin(dlat/2)**2 + cos(lat_rad) * cos(lat_inner_rad) * sin(dlong/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                distance = 6371 * c  # Earth radius is approximately 6371 km

                # Check if the earthquake is within the specified radius
                if distance <= radius:
                    magnitudes_within_radius.append(magnitude_inner)

            # Calculate the mean magnitude for earthquakes within the radius
            mean_magnitude = pd.Series(magnitudes_within_radius).mean()
            mean_magnitudes.append(mean_magnitude)
            
            pbar.update(1)

    df['Mean Magnitude'] = pd.Series(mean_magnitudes)
    
    return df

