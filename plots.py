import numpy as np
import matplotlib.pyplot as plt


def plot_raw_sensors(time_s, mx, my, speed, yawrate):
    """Raw readings of the sensors used by the filter."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Raw Sensor Readings', fontsize=13)

    axes[0].plot(time_s, mx, label='GPS X — East [m]')
    axes[0].plot(time_s, my, label='GPS Y — North [m]')
    axes[0].set_ylabel('Position [m]')
    axes[0].set_title('GPS Position')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, speed / 3.6, label='Speed [m/s]')
    axes[1].set_ylabel('Speed [m/s]')
    axes[1].set_title('GPS Speed')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_s, yawrate, label='Yaw Rate [deg/s]')
    axes[2].set_ylabel('Yaw Rate [deg/s]')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_title('IMU Yaw Rate')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()


def plot_vehicle_speed(steps, x3):
    """
    Longitudinal and lateral vehicle speed in the body frame.
    CTRV assumes no lateral slip, so v_lat = 0 by model definition.
    """
    v_long = x3
    v_lat  = np.zeros_like(x3)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.step(steps, v_long, label='Longitudinal $v_{long}$ [m/s]')
    ax.step(steps, v_lat,  label='Lateral $v_{lat}$ [m/s]  (CTRV: no sideslip)',
            linestyle='--', alpha=0.6)
    ax.set_xlabel('Filter Step')
    ax.set_ylabel('Speed [m/s]')
    ax.set_title('Vehicle Speed — Body Frame')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_heading(steps, x2):
    """UKF-estimated vehicle heading."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(steps, x2, label='Heading $\\psi$ [rad]')
    ax.set_xlabel('Filter Step')
    ax.set_ylabel('Heading [rad]')
    ax.set_title('Vehicle Heading')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_angular_velocity(steps, x4):
    """UKF-estimated vehicle angular velocity (yaw rate)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(steps, x4, label='Yaw rate $\\dot{\\psi}$ [rad/s]')
    ax.set_xlabel('Filter Step')
    ax.set_ylabel('Yaw Rate [rad/s]')
    ax.set_title('Vehicle Angular Velocity')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_position(x0, x1, x2, mx, my, GPS):
    """UKF position trajectory overlaid with raw GPS fixes."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(x0, x1, 'k-', lw=3, label='UKF Position')
    ax.quiver(x0[::20], x1[::20],
              np.cos(x2[::20]), np.sin(x2[::20]),
              color='#94C600', units='xy', width=0.5, scale=0.1,
              label='Heading')
    ax.scatter(mx[GPS], my[GPS], s=30, marker='+', label='GPS Measurements')
    ax.scatter(x0[0],  x1[0],  s=80, c='g', zorder=5, label='Start')
    ax.scatter(x0[-1], x1[-1], s=80, c='r', zorder=5, label='Goal')
    ax.set_xlabel('X [m]  (East)')
    ax.set_ylabel('Y [m]  (North)')
    ax.set_title('UKF — Vehicle Position')
    ax.legend(loc='best')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
