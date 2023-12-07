import numpy as np

def resample(particles, particles_w):
    """This function should resample the particles based on their weights (eq. 6.), and return these new
    particles along with their corresponding weights.
    
    Returns:
        resampled_particles: num_particles x 2 or num_particles x 4
        resampled_particles_w: num_particles x 1
    """
    print(f"weights sum: {np.sum(particles_w)}")
    # Resample particles based on their weights
    resampled_particles = []
    resampled_particles_w = []
    for _ in range(len(particles)):
        # Select a particle based on the weights
        idx = np.random.choice(len(particles), p=particles_w.flatten()/np.sum(particles_w), replace=True)
        resampled_particles.append(particles[idx])
        resampled_particles_w.append(particles_w[idx])
    
    resampled_particles = np.array(resampled_particles)
    resampled_particles_w = np.array(resampled_particles_w).reshape(-1, 1)
    
    return resampled_particles, resampled_particles_w