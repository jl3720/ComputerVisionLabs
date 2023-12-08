import numpy as np
from chi2_cost import chi2_cost
from color_histogram import color_histogram

def observe(particles, frame, bbox_height, bbox_width,
            hist_bin, hist, sigma_observe):
    """
    This function should make observations i.e. compute for all particles its color histogram describing
    the bounding box defined by the center of the particle and bbox height and bbox width. These
    observations should be used then to update the weights particles w using eq. 6 based on the χ2
    distance between the particle color histogram and the target color histogram given here as hist
    target. In order to compute the χ2 distance use the provided function chi2 cost.py.
    
    Returns:
        particles_w: num_particles x 1
    """
    particles_w = []
    for particle in particles:
        # Compute the bounding box coordinates
        x = int(particle[0] - bbox_width / 2)
        y = int(particle[1] - bbox_height / 2)
        x_end = int(particle[0] + bbox_width / 2)
        y_end = int(particle[1] + bbox_height / 2)

        hist_roi = color_histogram(x, y, x_end, y_end, frame, hist_bin)
        
        # Compute the chi2 distance between the particle histogram and the target histogram
        distance = chi2_cost(hist_roi, hist)
        
        # Compute the weight of the particle using the chi2 distance and sigma_observe
        weight = np.exp(-0.5 * (distance/sigma_observe)**2) / (np.sqrt(2*np.pi) * sigma_observe)

        particles_w.append(weight+np.nextafter(0, 1))  # add epsilon to avoid zero weight
    
    particles_w = np.array(particles_w).reshape(-1, 1)
    
    return particles_w
    