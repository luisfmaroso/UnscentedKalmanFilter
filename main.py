# ==============================================================
#  SIGMA POINT METHOD — choose one:
#    'merwe'            Van der Merwe scaled  (2n+1 points)
#    'julier'           Julier & Uhlmann      (2n+1 points)
#    'simplex'          Moireau & Chapelle    (n+1  points)
#    'spherical_radial' Cubature / CKF        (2n   points)
# ==============================================================
SIGMA_METHOD = 'merwe'
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sigma_points import (MerweScaledSigmaPoints, JulierSigmaPoints,
                           SimplexSigmaPoints, SphericalRadialSigmaPoints)
from ukf import ukf_predict, ukf_update, Hx_gps, Hx_nogps
from plots import (plot_raw_sensors, plot_vehicle_speed,
                   plot_heading, plot_angular_velocity, plot_position)


# ============================================================
# Load Data
# ============================================================

df = pd.read_csv('data/2014-03-26-000-Data.csv')

millis    = df['millis'].to_numpy()
yawrate   = df['yawrate'].to_numpy()
speed     = df['speed'].to_numpy()
course    = df['course'].to_numpy()
latitude  = df['latitude'].to_numpy()
longitude = df['longitude'].to_numpy()
altitude  = df['altitude'].to_numpy()

print(f"Read 'data/2014-03-26-000-Data.csv' successfully. Samples: {len(millis)}")

# Course: GPS convention 0=North CW → math convention 0=East CCW [deg]
course = -course + 90.0

# Convert lat/lon to Cartesian metres
RadiusEarth = 6378388.0
arc = 2.0 * np.pi * (RadiusEarth + altitude) / 360.0   # m/deg

dx = arc * np.cos(latitude * np.pi / 180.0) * np.hstack((0.0, np.diff(longitude)))
dy = arc * np.hstack((0.0, np.diff(latitude)))

mx  = np.cumsum(dx)     # East  position [m]
my  = np.cumsum(dy)     # North position [m]
ds  = np.sqrt(dx**2 + dy**2)
GPS = (ds != 0.0).astype(bool)

print(f"GPS updates: {np.sum(GPS)} / {len(GPS)} steps")


# ============================================================
# Filter Parameters
# ============================================================

numstates = 5
dt        = 1.0 / 50.0     # IMU sample rate [s]

sGPS      = 0.5 * 8.8 * dt**2
sCourse   = 0.1 * dt
sVelocity = 8.8 * dt
sYaw      = 1.0 * dt

Q = np.diag([sGPS**2, sGPS**2, sCourse**2, sVelocity**2, sYaw**2])

varGPS   = 6.0
varspeed = 1.0
varyaw   = 0.1

R_gps   = np.diag([varGPS**2, varGPS**2, varspeed**2, varyaw**2])
R_nogps = np.diag([varspeed**2, varyaw**2])


# ============================================================
# Sigma Point Selection
# ============================================================

if SIGMA_METHOD == 'merwe':
    points = MerweScaledSigmaPoints(n=numstates, alpha=1e-3, beta=2.0, kappa=0.0)
elif SIGMA_METHOD == 'julier':
    points = JulierSigmaPoints(n=numstates, kappa=0.0)
elif SIGMA_METHOD == 'simplex':
    points = SimplexSigmaPoints(n=numstates)
elif SIGMA_METHOD == 'spherical_radial':
    points = SphericalRadialSigmaPoints(n=numstates)
else:
    raise ValueError(f"Unknown SIGMA_METHOD: {SIGMA_METHOD!r}. "
                     "Choose 'merwe', 'julier', 'simplex', or 'spherical_radial'.")

print(f"Sigma method: {SIGMA_METHOD}  ({points.num_sigmas()} sigma points)")


# ============================================================
# Initial State  (bootstrapped from first GPS fix)
# ============================================================

x = np.array([
    mx[0],
    my[0],
    course[0] / 180.0 * np.pi,
    speed[0]  / 3.6 + 0.001,       # km/h → m/s  (+tiny offset avoids zero)
    yawrate[0] / 180.0 * np.pi,    # deg/s → rad/s
])

P = np.diag([
    1000.0,         # x position    [m²]      — large: GPS-level initial uncertainty
    1000.0,         # y position    [m²]
    np.pi**2,       # heading       [rad²]    — π rad std dev = maximum for a bounded angle
    100.0,          # speed         [m²/s²]   — ~10 m/s std dev
    1.0,            # yaw rate      [rad²/s²] — ~1 rad/s std dev
])

measurements = np.vstack((mx, my, speed / 3.6, yawrate / 180.0 * np.pi))
m_steps      = measurements.shape[1]


# ============================================================
# Storage
# ============================================================

x0, x1, x2, x3, x4 = [], [], [], [], []


# ============================================================
# UKF Filter Loop
# ============================================================

for filterstep in range(m_steps):

    x, P, sigmas_f = ukf_predict(x, P, Q, dt, points)

    if GPS[filterstep]:
        z       = measurements[:, filterstep]
        x, P, _ = ukf_update(x, P, sigmas_f, z, R_gps, Hx_gps, points)
    else:
        z       = measurements[2:, filterstep]
        x, P, _ = ukf_update(x, P, sigmas_f, z, R_nogps, Hx_nogps, points)

    x0.append(x[0]);  x1.append(x[1])
    x2.append(x[2]);  x3.append(x[3]);  x4.append(x[4])


print("Filter loop complete.")

time_s = (millis - millis[0]) / 1000.0
steps  = np.arange(m_steps)

x0 = np.array(x0);  x1 = np.array(x1)
x2 = np.array(x2);  x3 = np.array(x3);  x4 = np.array(x4)


# ============================================================
# Plots
# ============================================================

plot_raw_sensors(time_s, mx, my, speed, yawrate)
plot_vehicle_speed(steps, x3)
plot_heading(steps, x2)
plot_angular_velocity(steps, x4)
plot_position(x0, x1, x2, mx, my, GPS)

plt.show()
