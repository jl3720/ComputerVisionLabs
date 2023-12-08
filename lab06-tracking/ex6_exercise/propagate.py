import numpy as np

def propagate(particles, frame_height, frame_width, params):
    """ 
    This function should propagate the particles given the system prediction model (matrix A) and the
    system model noise represented by params.model, params.sigma position and params.sigma velocity.
    Use the parameter frame height and frame width to make sure that the center of the particle lies
    inside the frame.

    Parameters:
        particles: num_particles x 2 or num_particles x 4
        frame_height: height of the frame
        frame_width: width of the frame
        params: parameters

    Returns:
        propagated_particles: num_particles x 2 or num_particles x 4
    """
    if params["model"] == 0:
        # No motion model
        A = np.eye(2)
    elif params["model"] == 1:
        # Constant velocity model
        A = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
    sigma_position = params["sigma_position"]  # System model noise for position
    sigma_velocity = params["sigma_velocity"]  # System model noise for velocity

    # Propagate particles using the prediction model and system model noise
    num_particles = len(particles)
    noise_position = np.random.normal(0, sigma_position, size=(num_particles, 2))
    noise_velocity = np.random.normal(0, sigma_velocity, size=(num_particles, 2))

    # Propagate particles: s_t = As_{t-1} + w_{t-1}
    if params["model"] == 0:
        propagated_particles = np.matmul(particles, A.T) + noise_position
    elif params["model"] == 1:
        propagated_particles = np.matmul(particles, A.T) + np.hstack((noise_position, noise_velocity))

    # Ensure particles stay within the frame
    propagated_particles[:, 0] = np.clip(propagated_particles[:, 0], 0, frame_width - 1)
    propagated_particles[:, 1] = np.clip(propagated_particles[:, 1], 0, frame_height - 1)

    return propagated_particles
    