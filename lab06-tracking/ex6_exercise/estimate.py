import numpy as np

def estimate(particles, particles_w: np.ndarray):
    """
    This function should estimate the mean state 
    given the particles and their weights.

    Parameters:
        particles: num_particles x 2 or num_particles x 4
        particles_w: num_particles x 1

    Returns:
        mean_state: 1 x 2 or 1 x 4
    """
    # mean_state = np.average(particles, weights=particles_w.flatten(), axis=0)
    # print(f"mean state: {mean_state}")

    weighted_particles = particles * particles_w
    mean_state = np.sum(weighted_particles, axis=0) / np.sum(particles_w)
    print(f"mean state: {mean_state}")
    return mean_state
