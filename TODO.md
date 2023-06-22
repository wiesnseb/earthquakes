1. Temporal Features:
    * Extract year, month, day, and day of the week from the date column. These can help capture any seasonality or periodic patterns.
    * Calculate the time difference between each earthquake and the previous one to capture the inter-event duration.
2. Geographical Features:
    * Combine latitude and longitude to create a location feature, such as a cluster or region identifier.
    * Calculate the distance to known fault lines or tectonic plate boundaries.
3. Magnitude Features:
    * Calculate the average magnitude of earthquakes that occurred within a certain radius or time window from each data point.
    * Calculate the maximum, minimum, and standard deviation of magnitudes within a certain radius or time window.
4. Depth Features:
    * Create categorical depth bins based on predefined ranges (e.g., shallow, intermediate, deep).
    * Calculate the average depth of earthquakes that occurred within a certain radius or time window from each data point.
5. Historical Features:
    * Count the number of earthquakes that occurred within a certain radius or time window from each data point.
    * Calculate the cumulative magnitude or average magnitude change within a certain time window leading up to each earthquake.
6. Find dataset with constant movement