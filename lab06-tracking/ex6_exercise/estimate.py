import numpy as np

def estimate(particles, particles_w):
    """
    This function should estimate the mean state 
    given the particles and their weights.

    Returns:
        mean_state: 1 x 2 or 1 x 4
    """
    return particles * particles_w / np.sum(particles_w)
