import pandas as pd
import folium
from geopy.geocoders import Nominatim

from earthquakes.__main__ import df

# Select the last 50 earthquakes
latest_earthquakes = df.head(1000)

# Extract the latitude, longitude, and magnitude columns
latitude = latest_earthquakes['Latitude']
longitude = latest_earthquakes['Longitude']
magnitude = latest_earthquakes['Magnitude']

# Create a folium map centered on a specific location
map = folium.Map(location=[38.36, 31.17], zoom_start=6.5)

# Iterate over the latest earthquakes and add markers to the map
for lat, lon, mag in zip(latitude, longitude, magnitude):
    popup_text = f"Magnitude: {mag}"
    folium.CircleMarker(
        location=[lat, lon],
        radius=mag*5,
        color='red',
        fill=True,
        fill_color='red',
        popup=popup_text
    ).add_to(map)

map.save('index.html')

# in my python file i want to use a function located in a folder plots named def_test