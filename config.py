# config.py
"""
Contains all global constants and parameters for the scattering simulation.
"""
import numpy as np

# --- Scene and Scattering Parameters ---
ATMOSPHERE_RADIUS = 8.0
SUN_DISTANCE = 15.0
SUN_RADIUS = 1.5


# --- Physically-based scattering model ---
SUN_COLOR = np.array([1.0, 1.0, 1.0])  # Start with pure white light
NIGHT_COLOR = np.array([0.1, 0.1, 0.15])

GODRAY = SUN_COLOR * 1.5  # Multiplier for godray intensity
# Coefficients control how much of each color is scattered.
# Blue scatters the most, Green less, Red the least (approximating 1/λ^4).
SCATTERING_COEFFICIENTS = np.array([0.1, 0.4, 0.9])  # R, G, B
SCATTERING_STRENGTH = 0.25  # An overall multiplier to tune the effect
MAX_COLOR_ENHANCEMENT = 100.0  # Maximum multiplication factor for red/green enhancement
EXPONENTIAL_COEFFICIENT = -162.0  # Coefficient for exponential enhancement curve
NUM_SEGMENTS = 15  # The resolution of the simulation along the ray path