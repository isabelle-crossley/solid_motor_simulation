# -*- coding: utf-8 -*-
"""
Created on Fri May 30 12:25:06 2025

@author: Isabelle Crossley

star grain regression  code, could easily add different shape mask functions to
simulate regression of other core shapes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import disk

def precompute_polar_grid(shape):
    """
    creates normalised cartesian and polar coordinate grids for a given image shape

    Parameters:
        shape (tuple): (height, width) in pixels

    Returns:
        x, y: normalized cartesian coordinates in range [-1, 1]
        r: radial distance from center (0 = center, 1 = edge)
        theta: angle in radians, from -pi to pi
        """
    h, w = shape
    y, x = np.indices((h, w))
    x = (x - w // 2) / (w // 2)
    y = (y - h // 2) / (h // 2)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return x, y, r, theta

def create_curvy_star_grain_mask(shape, r_core0=0.2, r_tip0=0.45, n_lobes=6, lobe_depth=0.1):
    """
    creates a boolean mask for an initial curvy star grain geometry
    
    the radius at each angle oscillates sinusoidally to create lobes
    
    Parameters:
        r_core0: Base core radius (fraction of casing radius)
        r_tip0: Maximum radius of the lobe tips (fraction of casing radius)
        n_lobes: Number of star points/lobes
        lobe_depth: Amplitude of the sinusoidal variation in radius
    
    Returns:
        mask (bool array): True inside propellant, False outside
    """
    h, w = shape
    x, y, r, theta = precompute_polar_grid(shape)
    r_lobe = r_core0 + lobe_depth * (1 + np.sin(n_lobes * theta))
    r_lobe = np.clip(r_lobe, None, r_tip0)
    mask = r <= r_lobe
    return mask

def simulate_outward_burn_rounded(r_core0=0.2, r_tip0=0.5, r_dot=0.01, total_time=4.0, dt=0.5,
                                  casing_radius=1.0, n_spokes=6, groove_width=np.pi/18,
                                  resolution=512):
    """
    simulates and plots the grain burn progression over time
    
    uses a distance transform to avoid simulating burn layer-by-layer,
    allowing fast calculation of burn fronts
    
    Parameters:
        r_core0: Initial core radius
        r_tip0: Initial tip radius
        r_dot: Burn rate (normalised units per second)
        total_time: Total burn time (s)
        dt: Time step for plotting (s)
        casing_radius: Outer radius of the motor casing
        n_spokes, groove_width: (Currently unused) for future spoke features
        resolution: Number of pixels in the simulation grid
    """
    shape = (resolution, resolution)
    pixel_size = 2.0 / resolution  # normalized space [-1, 1]
    times = np.arange(0, total_time + dt, dt)

    # Create coordinate grid for casing mask
    y, x = np.indices(shape)
    x = (x - shape[1] // 2) / (shape[1] // 2)
    y = (y - shape[0] // 2) / (shape[0] // 2)
    r_grid = np.sqrt(x**2 + y**2)
    casing_mask = r_grid <= casing_radius

    # Create initial grain mask
    mask = create_curvy_star_grain_mask(shape, r_core0=0.2, r_tip0=0.45, n_lobes=6, lobe_depth=0.1)

    # Compute distance transform once
    inverted_mask = ~mask
    distance = distance_transform_edt(inverted_mask)

    # Start plotting
    plt.figure(figsize=(8, 8))

    for t in times:
        burn_radius = r_dot * t
        pixel_radius = burn_radius / pixel_size
        burned = distance <= pixel_radius

        # Clip burned area to within casing
        burned &= casing_mask

        plt.contour(burned.astype(float), levels=[0.5],
                    extent=[-1, 1, -1, 1], colors='black', linewidths=1)

    # Draw casing
    circle = plt.Circle((0, 0), casing_radius, color='k', linestyle='--', fill=False, label='Casing')
    plt.gca().add_patch(circle)

    plt.axis("equal")
    plt.title("Star Grain Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

#run the sim time!

simulate_outward_burn_rounded(
    r_core0=0.2,
    r_tip0=0.5,
    r_dot=0.05,
    total_time=15.0,
    dt=1.0,
    casing_radius=1.0,
    n_spokes=6,
    groove_width=np.pi/18,
    resolution=512
)
