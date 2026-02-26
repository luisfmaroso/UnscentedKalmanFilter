import numpy as np


# ============================================================
# Utility
# ============================================================

def normalize_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ============================================================
# CTRV Motion Model  (Fx)
#
# State: x = [x_pos, y_pos, psi, v, psi_dot]
#   x_pos   : position East  [m]
#   y_pos   : position North [m]
#   psi     : heading        [rad]
#   v       : speed          [m/s]
#   psi_dot : yaw rate       [rad/s]
# ============================================================

def Fx(x, dt):
    """CTRV state transition function applied to a single state vector."""
    x_new   = x.copy()
    psi     = x[2]
    v       = x[3]
    psi_dot = x[4]

    if abs(psi_dot) < 0.0001:           # straight-line motion
        x_new[0] = x[0] + v * dt * np.cos(psi)
        x_new[1] = x[1] + v * dt * np.sin(psi)
        x_new[2] = psi
        x_new[3] = v
        x_new[4] = 1e-7                 # keep non-zero to avoid numerical issues
    else:                               # curved motion
        x_new[0] = x[0] + (v / psi_dot) * (np.sin(psi_dot * dt + psi) - np.sin(psi))
        x_new[1] = x[1] + (v / psi_dot) * (-np.cos(psi_dot * dt + psi) + np.cos(psi))
        x_new[2] = psi + psi_dot * dt  # no per-point wrap — preserve arithmetic centroid
        x_new[3] = v
        x_new[4] = psi_dot
    return x_new


# ============================================================
# Measurement Functions  (Hx)
# ============================================================

def Hx_gps(x):
    """Maps state to measurement when GPS is available: [x, y, v, psi_dot]."""
    return np.array([x[0], x[1], x[3], x[4]])


def Hx_nogps(x):
    """Maps state to measurement when GPS is not available: [v, psi_dot]."""
    return np.array([x[3], x[4]])


# ============================================================
# UKF Prediction Step
# ============================================================

def ukf_predict(x, P, Q, dt, points):
    """
    UKF time-update (prediction).

    Parameters
    ----------
    x      : state mean (n,)
    P      : state covariance (n, n)
    Q      : process noise covariance (n, n)
    dt     : time step [s]
    points : sigma point object (any of the four classes in sigma_points.py)

    Returns
    -------
    x_pred   : predicted state mean (n,)
    P_pred   : predicted state covariance (n, n)
    sigmas_f : propagated sigma points (num_sigmas, n)
    """
    sigmas   = points.sigma_points(x, P)
    n_sig    = sigmas.shape[0]

    sigmas_f = np.array([Fx(sigmas[i], dt) for i in range(n_sig)])

    # Predicted mean
    # Arithmetic weighted sum — correct for all four sigma point methods:
    #   • Symmetric methods (Julier, Spherical Radial, Merwe): symmetric pairs
    #     cancel exactly, giving x + psi_dot·dt for the heading component.
    #   • Simplex: the arithmetic centroid of the sigma points equals x by
    #     construction, so the weighted sum also gives x + psi_dot·dt exactly.
    # A single normalize_angle on the final mean handles the ±π wrap cleanly.
    # Per-point normalization inside Fx is intentionally avoided: it destroys
    # the centroid property and caused the atan2/circular mean to be needed.
    x_pred    = np.dot(points.Wm, sigmas_f)
    x_pred[2] = normalize_angle(x_pred[2])

    # Predicted covariance
    P_pred = Q.copy()
    for i in range(n_sig):
        diff    = sigmas_f[i] - x_pred
        diff[2] = normalize_angle(diff[2])
        P_pred += points.Wc[i] * np.outer(diff, diff)

    P_pred = 0.5 * (P_pred + P_pred.T)     # symmetrise

    return x_pred, P_pred, sigmas_f


# ============================================================
# UKF Update Step
# ============================================================

def ukf_update(x_pred, P_pred, sigmas_f, z, R, hx_func, points):
    """
    UKF measurement-update (correction).

    Parameters
    ----------
    x_pred   : predicted state mean (n,)
    P_pred   : predicted state covariance (n, n)
    sigmas_f : propagated sigma points from prediction step (num_sigmas, n)
    z        : actual measurement vector (m,)
    R        : measurement noise covariance (m, m)
    hx_func  : measurement function  z_hat = hx_func(x)
    points   : sigma point object

    Returns
    -------
    x_new : updated state mean (n,)
    P_new : updated state covariance (n, n)
    K     : Kalman gain (n, m)
    """
    n_sig   = sigmas_f.shape[0]
    n_state = x_pred.shape[0]
    n_meas  = z.shape[0]

    sigmas_h = np.array([hx_func(sigmas_f[i]) for i in range(n_sig)])

    z_pred = np.dot(points.Wm, sigmas_h)

    Pzz = R.copy()
    Pxz = np.zeros((n_state, n_meas))
    for i in range(n_sig):
        dz      = sigmas_h[i] - z_pred
        dx      = sigmas_f[i] - x_pred
        dx[2]   = normalize_angle(dx[2])
        Pzz    += points.Wc[i] * np.outer(dz, dz)
        Pxz    += points.Wc[i] * np.outer(dx, dz)

    K          = Pxz @ np.linalg.inv(Pzz)
    x_new      = x_pred + K @ (z - z_pred)
    x_new[2]   = normalize_angle(x_new[2])
    P_new      = P_pred - K @ Pzz @ K.T
    P_new      = 0.5 * (P_new + P_new.T)   # symmetrise

    return x_new, P_new, K
