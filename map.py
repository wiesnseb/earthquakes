import pandas as pd
import folium
from geopy.geocoders import Nominatim

from main import eq

# Select the last 50 earthquakes
latest_earthquakes = eq.head(1000)

# Extract the latitude, longitude, and magnitude columns
latitude = latest_earthquakes['Latitude']
longitude = latest_earthquakes['Longitude']
magnitude = latest_earthquakes['Mag']

# Create a folium map centered on a specific location
map = folium.Map(location=[0, 0], zoom_start=3)

# Iterate over the latest earthquakes and add markers to the map
for lat, lon, mag in zip(latitude, longitude, magnitude):
    popup_text = f"Magnitude: {mag}"
    folium.CircleMarker(
        location=[lat, lon],
        radius=mag,
        color='red',
        fill=True,
        fill_color='red',
        popup=popup_text
    ).add_to(map)


map.save('index.html')
