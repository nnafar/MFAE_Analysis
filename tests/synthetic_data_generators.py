# -*- coding: utf-8 -*-
"""
Generates synthetic test images with known ground truth.

WHY THIS MATTERS:
Real cell images have unknown "true" values. Synthetic images let us test
if the algorithm can recover parameters we deliberately set.

For example: If we draw a protrusion that's EXACTLY 10 pixels long,
does the detector measure ~10 pixels? If not, the algorithm is wrong.
"""

import numpy as np
import cv2
from typing import Tuple


def create_synthetic_protrusion_image(
    width: int = 200,
    height: int = 50,
    protrusion_length_px: float = 30.0,
    pipette_x: int = 150,
    cell_brightness: int = 180,
    background_brightness: int = 50,
    noise_level: float = 10.0,
    blur_sigma: float = 1.0
) -> Tuple[np.ndarray, dict]:
    """
    Creates a synthetic image of a cell being aspirated into a pipette.
    
    **How it works:**
    1. Creates a black canvas
    2. Draws a bright rectangle (the "cell body") on the right side
    3. Draws a bright rectangle extending into the "pipette" (the "protrusion")
    4. Adds realistic noise (Gaussian blur + random noise)
    5. Returns the image + a dictionary with the TRUE values we used
    
    Args:
        width: Image width in pixels (horizontal dimension)
        height: Image height in pixels (vertical dimension)
        protrusion_length_px: How far the cell extends into the pipette (in pixels)
                               This is our GROUND TRUTH that we'll test against
        pipette_x: X-coordinate of the pipette entrance (left edge)
        cell_brightness: Grayscale intensity of the cell (0-255)
        background_brightness: Grayscale intensity of empty channel (0-255)
        noise_level: Standard deviation of Gaussian noise to add (realistic noise ~5-15)
        blur_sigma: Amount of Gaussian blur to apply (simulates optical diffraction)
    
    Returns:
        Tuple of:
            - image: A numpy array (height x width) with dtype uint8
            - ground_truth: Dictionary with keys:
                - 'protrusion_length_px': The TRUE length we drew
                - 'pipette_x': Where the pipette entrance is
                - 'cell_left_edge': X-coordinate of the protrusion tip
                - 'cell_brightness': Mean intensity of the cell region
    
    Example:
        >>> img, truth = create_synthetic_protrusion_image(protrusion_length_px=25.5)
        >>> print(truth['protrusion_length_px'])  # Output: 25.5
        >>> # Now test if your detector recovers ~25.5 from the image
    """
    # 1. Create blank canvas (all background brightness)
    image = np.full((height, width), background_brightness, dtype=np.float32)
    
    # 2. Calculate geometry
    # The protrusion starts at the pipette entrance and extends LEFT
    protrusion_right = pipette_x  # Right edge is at the pipette
    protrusion_left = int(pipette_x - protrusion_length_px)  # Left edge
    
    # The cell body starts where the protrusion ends and extends RIGHT to edge
    body_left = pipette_x
    body_right = width
    
    # 3. Draw the protrusion (inside the pipette)
    # Make it slightly narrower than full height to simulate realistic cell shape
    margin = int(height * 0.25)  # Top and bottom margin (25% of height)
    image[margin:height-margin, protrusion_left:protrusion_right] = cell_brightness
    
    # 4. Draw the cell body (outside the pipette, to the right)
    image[margin:height-margin, body_left:body_right] = cell_brightness
    
    # 5. Add realistic noise
    # Gaussian noise simulates camera sensor noise
    noise = np.random.normal(0, noise_level, image.shape)
    image = image + noise
    
    # 6. Apply Gaussian blur (simulates optical diffraction/defocus)
    if blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    
    # 7. Clip to valid range and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # 8. Package the ground truth
    ground_truth = {
        'protrusion_length_px': protrusion_length_px,
        'pipette_x': pipette_x,
        'cell_left_edge': protrusion_left,
        'cell_brightness': cell_brightness,
        'background_brightness': background_brightness
    }
    
    return image, ground_truth


def create_synthetic_rupture_trace(
    n_frames: int = 100,
    baseline_mean: float = 100.0,
    baseline_noise: float = 2.0,
    rupture_frame: int = 50,
    post_rupture_mean: float = 130.0,
    transition_frames: int = 3
) -> Tuple[np.ndarray, dict]:
    """
    Creates a synthetic "intensity vs time" trace with a rupture event.
    
    **Use case:**
    Testing the CUSUM rupture detection algorithm. We create a signal where
    intensity suddenly jumps (simulating cytoplasm leaking), then check if
    CUSUM detects it at the correct frame.
    
    Args:
        n_frames: Total number of time points
        baseline_mean: Average intensity before rupture (arbitrary units)
        baseline_noise: Standard deviation of noise before rupture
        rupture_frame: Frame index where rupture occurs (0-indexed)
        post_rupture_mean: Average intensity after rupture (must be > baseline)
        transition_frames: How many frames the rupture takes to develop
                           (1 = instant jump, 5 = gradual increase)
    
    Returns:
        Tuple of:
            - trace: 1D numpy array of intensity values
            - ground_truth: Dict with 'rupture_frame' and 'rupture_detected' (always True)
    
    Example:
        >>> trace, truth = create_synthetic_rupture_trace(rupture_frame=50)
        >>> # Now pass trace to your CUSUM detector
        >>> detected, detected_frame = cusum_detect(trace)
        >>> assert detected == True
        >>> assert abs(detected_frame - 50) < 5  # Should detect within ±5 frames
    """
    # 1. Create baseline (pre-rupture) with Gaussian noise
    baseline = np.random.normal(baseline_mean, baseline_noise, rupture_frame)
    
    # 2. Create transition (rupture happening)
    # Linear ramp from baseline_mean to post_rupture_mean
    transition = np.linspace(baseline_mean, post_rupture_mean, transition_frames)
    
    # 3. Create post-rupture with same noise level
    remaining_frames = n_frames - rupture_frame - transition_frames
    post_rupture = np.random.normal(post_rupture_mean, baseline_noise, remaining_frames)
    
    # 4. Concatenate all parts
    trace = np.concatenate([baseline, transition, post_rupture])
    
    # 5. Package ground truth
    ground_truth = {
        'rupture_frame': rupture_frame,
        'rupture_detected': True,  # We know rupture happened (we created it!)
        'baseline_mean': baseline_mean,
        'post_rupture_mean': post_rupture_mean
    }
    
    return trace, ground_truth


def create_synthetic_time_series(
    n_frames: int = 100,
    model: str = 'jeffreys',
    E: float = 3000.0,      # Elastic modulus (Pa)
    eta1: float = 5000.0,   # Viscosity 1 (Pa·s)
    eta2: float = 15000.0,  # Viscosity 2 (Pa·s)
    r_eff: float = 5.0,     # Effective radius (μm)
    delta_p: float = 1000.0,  # Pressure (Pa)
    C: float = 1.0,
    noise_level_um: float = 0.2,  # Measurement noise (μm)
    frame_interval: float = 1.0   # Time between frames (seconds)
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Creates synthetic protrusion length data following a known viscoelastic model.
    
    **Use case:**
    Testing the fitting algorithm. We generate "perfect" data from known parameters,
    then check if the fitter can recover those parameters.
    
    Args:
        n_frames: Number of time points to generate
        model: Which model to use ('jeffreys', 'kelvin_voigt', 'burgers')
        E, eta1, eta2: Model parameters (see Calculation_MFA.py for definitions)
        r_eff: Effective channel radius (μm)
        delta_p: Applied pressure (Pa)
        C: Geometric correction factor
        noise_level_um: Std dev of Gaussian noise to add to lengths (μm)
        frame_interval: Time step (seconds)
    
    Returns:
        Tuple of:
            - time: 1D array of time points (seconds)
            - lengths: 1D array of protrusion lengths (μm) with noise
            - ground_truth: Dict with all the TRUE parameters we used
    
    Example:
        >>> time, lengths, truth = create_synthetic_time_series(E=3000, eta1=5000, eta2=15000)
        >>> # Now fit the data
        >>> fitter = FittingMFA(time, lengths, r_eff=5.0, delta_p=1000)
        >>> fitter.run()
        >>> # Check if fitted E is close to 3000
        >>> fitted_E = fitter.best_fit_params['parameters']['E']
        >>> assert abs(fitted_E - 3000) / 3000 < 0.10  # Within 10% error
    """
    # Import here to avoid circular dependency
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from Calculation_MFA import jeffreys_length, kelvin_voigt_length, burgers_length
    
    # 1. Create time array
    time = np.arange(n_frames) * frame_interval
    
    # 2. Generate "perfect" lengths from the model
    if model.lower() == 'jeffreys':
        lengths_perfect = jeffreys_length(time, r_eff, delta_p, C, E, eta1, eta2)
    elif model.lower() == 'kelvin_voigt':
        lengths_perfect = kelvin_voigt_length(time, r_eff, delta_p, C, E, eta1)
    elif model.lower() == 'burgers':
        # For Burgers, use E1=E, E2=E (same stiffness), eta1, eta2 as given
        lengths_perfect = burgers_length(time, r_eff, delta_p, C, E, eta1, E, eta2)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # 3. Add realistic measurement noise
    noise = np.random.normal(0, noise_level_um, n_frames)
    lengths_noisy = lengths_perfect + noise
    
    # Ensure no negative lengths (physically impossible)
    lengths_noisy = np.maximum(lengths_noisy, 0)
    
    # 4. Package ground truth
    ground_truth = {
        'model': model,
        'E': E,
        'eta1': eta1,
        'eta2': eta2,
        'r_eff': r_eff,
        'delta_p': delta_p,
        'C': C,
        'noise_level_um': noise_level_um
    }
    
    return time, lengths_noisy, ground_truth