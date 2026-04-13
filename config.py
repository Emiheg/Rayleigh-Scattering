# config.py
"""
Contains all global constants and parameters for the scattering simulation.
"""
import numpy as np

# --- Scene and Scattering Parameters ---
ATMOSPHERE_RADIUS = 10
SUN_DISTANCE = 15
SUN_RADIUS = 2

NUM_SEGMENTS = 25   # The resolution of the simulation along the ray path
NUM_RAYS = 10      # Number of rays to simulate across the sun - for preview purposes only, not used on canvas
NUM_LINE_POINTS = 50  # Number of points to sample along the line for color sampling

SEGMENT_LENGTH = (ATMOSPHERE_RADIUS*2) / NUM_SEGMENTS

# --- Physically-based scattering model ---
SUN_COLOR = np.array([1.0, 1.0, 1.0])  # Start with pure white light

GODRAY = SUN_COLOR * 2  # Multiplier for godray intensity

# Coefficients control how much of each color is scattered.
# Blue scatters the most, Green less, Red the least (approximating 1/λ^4).
SCATTERING_COEFFICIENTS = np.array([0.1, 0.4, 0.9])  # R, G, B
SCATTERING_STRENGTH = 0.15  # An overall multiplier to tune the effect
MAX_COLOR_ENHANCEMENT = 100.0  # Maximum multiplication factor for red/green enhancement
EXPONENTIAL_COEFFICIENT = -162.0  # Coefficient for exponential enhancement curve
