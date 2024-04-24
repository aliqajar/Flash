

import numpy as np

def shirley_background(intensities, max_iter=100, tolerance=1e-5):
    """
    Simplified Shirley background removal.
    
    :param intensities: 1D array of intensities in the spectrum.
    :param max_iter: Maximum number of iterations.
    :param tolerance: Convergence tolerance.
    :return: Estimated background.
    """
    background = np.zeros_like(intensities)
    old_background = background.copy()
    
    for _ in range(max_iter):
        for i in range(1, len(intensities) - 1):
            background[i] = np.trapz(intensities[i:] - background[i:], dx=1)
        background = background / np.max(background) * (intensities[-1] - intensities[0]) + intensities[0]
        
        # Check for convergence
        if np.max(np.abs(background - old_background)) < tolerance:
            break
        old_background = background.copy()
        
    return background
