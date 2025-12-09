import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data/2014-03-26-000-Data.csv')

# Extract data into numpy arrays
millis = df['millis'].to_numpy()
ax = df['ax'].to_numpy()
ay = df['ay'].to_numpy()
az = df['az'].to_numpy()
rollrate = df['rollrate'].to_numpy()
pitchrate = df['pitchrate'].to_numpy()
yawrate = df['yawrate'].to_numpy()
latitude = df['latitude'].to_numpy()
longitude = df['longitude'].to_numpy()
altitude = df['altitude'].to_numpy()

# Normalize time - subtract first value to get relative time in milliseconds
time_ms = millis - millis[0]
time_s = time_ms / 1000.0  # Convert to seconds for plotting

# Convert latitude/longitude to meters from origin
# Using the first point as origin (0, 0)
lat_origin = latitude[0]
lon_origin = longitude[0]

# Convert to meters using approximate conversion
# 1 degree latitude ≈ 111,111 meters
# 1 degree longitude ≈ 111,111 * cos(latitude) meters
lat_diff = latitude - lat_origin
lon_diff = longitude - lon_origin

y_meters = lat_diff * 111111.0  # North-South distance in meters
x_meters = lon_diff * 111111.0 * np.cos(np.radians(lat_origin))  # East-West distance in meters

# Altitude relative to first point
altitude_rel = altitude - altitude[0]

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Sensor Data vs Time', fontsize=16, fontweight='bold')

# Plot 1: Accelerometer data
axes[0].plot(time_s, ax, 'r-', label='ax', linewidth=1.5)
axes[0].plot(time_s, ay, 'g-', label='ay', linewidth=1.5)
axes[0].plot(time_s, az, 'b-', label='az', linewidth=1.5)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Acceleration (m/s²)')
axes[0].set_title('Accelerometer Readings')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Plot 2: Gyroscope data
axes[1].plot(time_s, rollrate, 'r-', label='Roll Rate', linewidth=1.5)
axes[1].plot(time_s, pitchrate, 'g-', label='Pitch Rate', linewidth=1.5)
axes[1].plot(time_s, yawrate, 'b-', label='Yaw Rate', linewidth=1.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Angular Rate (deg/s)')
axes[1].set_title('Gyroscope Readings')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

# Plot 3: GNSS data (position relative to origin)
axes[2].plot(time_s, x_meters, 'r-', label='X (East-West) [m]', linewidth=1.5)
axes[2].plot(time_s, y_meters, 'g-', label='Y (North-South) [m]', linewidth=1.5)
axes[2].plot(time_s, altitude_rel, 'b-', label='Altitude (relative) [m]', linewidth=1.5)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Position (m)')
axes[2].set_title('GNSS Position (relative to origin)')
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Print some statistics
print(f"Data Summary:")
print(f"Total duration: {time_s[-1]:.2f} seconds")
print(f"Number of samples: {len(time_s)}")
print(f"Origin coordinates: Lat={lat_origin:.6f}°, Lon={lon_origin:.6f}°, Alt={altitude[0]:.2f}m")
print(f"Max displacement: X={np.max(np.abs(x_meters)):.2f}m, Y={np.max(np.abs(y_meters)):.2f}m, Z={np.max(np.abs(altitude_rel)):.2f}m")