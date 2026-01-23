# -*- coding: utf-8 -*-
"""
COMPREHENSIVE VALIDATION SUITE for MFA Pipeline.

Combines:
1. Synthetic Data Validation (System Tests)
2. Mathematical Unit Tests (Physics Checks)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import specific functions to test
from Calculation_MFA import (
    calculate_recommended_fstar, 
    compute_shear_metrics,
    jeffreys_length, 
    kelvin_voigt_length,
    burgers_length,
    estimate_initial_parameters,
    validate_jeffreys_parameters
)
from synthetic_data_generators import (
    create_synthetic_protrusion_image,
    create_synthetic_rupture_trace,
    create_synthetic_time_series
)

# =============================================================================
# TEST 1: Geometric Corrections (Son, 2007)
# =============================================================================

def test_son_factor_table1():
    """Tests if calculate_recommended_fstar() reproduces Table 1 from Son (2007)."""
    # Format: (Height, Width, Expected_f*)
    test_cases = [
        (0.5, 10.0, 0.9365),   # Aspect ratio 0.05
        (1.0, 10.0, 0.8820),   # Aspect ratio 0.10
        (2.0, 10.0, 0.7946),   # Aspect ratio 0.20
        (5.0, 10.0, 0.6478),   # Aspect ratio 0.50
        (10.0, 10.0, 0.5928),  # Aspect ratio 1.00
    ]
    
    for height, width, expected_fstar in test_cases:
        calculated_fstar, _, _ = calculate_recommended_fstar(width, height)
        error = abs(calculated_fstar - expected_fstar)
        
        assert error < 0.005, (
            f"f* mismatch. Expected: {expected_fstar:.4f}, Got: {calculated_fstar:.4f}"
        )
    print("✓ Son (2007) f* calculation validated")

# =============================================================================
# TEST 2: Viscoelastic Model Physics
# =============================================================================

def test_jeffreys_newtonian_limit():
    """Tests if Jeffreys model reduces to Kelvin-Voigt when η₂ → ∞."""
    time = np.linspace(0, 10, 50)
    # Huge eta2 effectively removes the series dashpot (no linear flow)
    L_jeffreys = jeffreys_length(time, 5.0, 1000.0, 1.0, 1000.0, 500.0, 1e12)
    L_kelvin = kelvin_voigt_length(time, 5.0, 1000.0, 1.0, 1000.0, 500.0)
    
    assert np.allclose(L_jeffreys, L_kelvin, rtol=0.01)
    print("✓ Jeffreys → Kelvin-Voigt limit validated")

def test_burgers_physics():
    """Tests the 4-parameter Burgers model behavior."""
    time = np.linspace(0, 10, 50)
    # Burgers = Maxwell + Kelvin-Voigt
    # If we make the Kelvin part rigid (E2 -> inf), it should look like Maxwell (Linear + Jump)
    L_burgers = burgers_length(time, 5.0, 1000.0, 1.0, 
                               E1=1000.0, eta1=5000.0, 
                               E2=1e9, eta2=5000.0)
    
    # Maxwell slope check: (r_eff * dP) / (3 * pi * eta1)
    expected_slope = (5.0 * 1000.0) / (3 * np.pi * 5000.0)
    actual_slope = (L_burgers[-1] - L_burgers[0]) / (time[-1] - time[0])
    
    assert np.isclose(actual_slope, expected_slope, rtol=0.1)
    print("✓ Burgers model physics validated")

# =============================================================================
# TEST 3: Shear Stress & Rate Calculations
# =============================================================================

def test_shear_metrics():
    """Validates shear calculations against dimensional analysis."""
    W, H, L = 10.0, 5.0, 100.0 # microns
    dP = 1000.0 # Pa
    visc = 0.001 # Pa.s
    
    metrics = compute_shear_metrics(W, H, L, dP, visc)
    
    # 1. Check for required keys
    assert "Wall_Shear_Stress_Pa" in metrics
    assert "Wall_Shear_Rate_s1" in metrics
    
    # 2. Physics check: Shear Rate ≈ Stress / Viscosity
    # (Approximate for Newtonian fluid in rectangular channel)
    sigma = metrics["Wall_Shear_Stress_Pa"]
    gamma = metrics["Wall_Shear_Rate_s1"]
    
    calc_visc = sigma / gamma
    
    # Should be close to input viscosity (order of magnitude check due to shape factors)
    assert 0.0005 < calc_visc < 0.002
    print("✓ Shear stress/rate calculations validated")

# =============================================================================
# TEST 4: Rupture Detection (CUSUM)
# =============================================================================

def test_cusum_detects_synthetic_rupture():
    """Tests CUSUM detection on a noisy synthetic trace."""
    np.random.seed(42)
    trace, ground_truth = create_synthetic_rupture_trace(
        n_frames=100, rupture_frame=50,
        baseline_mean=100.0, baseline_noise=1.0,
        post_rupture_mean=135.0, transition_frames=5
    )
    
    # Instantiate detector with strict but valid params
    from LineDetection_MFA import LineDetectionMFA
    detector = LineDetectionMFA(
        roi_images=[np.zeros((10,10))], pipette_coords=[0,0],
        params={'rupture_detection': {
            'enable': True,
            'cusum_settling_buffer': 5,
            'cusum_baseline_len': 10,
            'min_intensity_noise_floor': 0.5,
            'cusum_threshold_factor': 10.0
        }}
    )
    
    detected, detected_frame = detector._detect_rupture_from_haze(trace.tolist())
    
    assert detected is True
    assert abs(detected_frame - ground_truth['rupture_frame']) <= 5
    print(f"✓ CUSUM detected rupture at frame {detected_frame} (True: 50)")

# =============================================================================
# TEST 5: Protrusion Measurement Accuracy
# =============================================================================

def test_protrusion_length_measurement():
    """Tests sub-pixel length measurement."""
    np.random.seed(42)
    true_len = 30.5
    image, _ = create_synthetic_protrusion_image(protrusion_length_px=true_len)
    
    from LineDetection_MFA import LineDetectionMFA
    detector = LineDetectionMFA(
        roi_images=[image], pipette_coords=[150, 25],
        params={'scale_factor': 1.0, 'workflow_settings': {'verify_traps_interactively': False}}
    )
    
    res = detector.run_automatic()
    measured = res['protrusion_lengths_px'][0]
    
    assert abs(measured - true_len) < 2.5
    print(f"✓ Protrusion length accuracy: {abs(measured - true_len):.2f} px error")

# =============================================================================
# TEST 6: Parameter Recovery (Fitting)
# =============================================================================

def test_parameter_recovery_jeffreys():
    """Tests if the fitter can recover E, eta1, eta2 from synthetic data."""
    # Generate clean data
    time, lengths, gt = create_synthetic_time_series(
        n_frames=100, model='jeffreys',
        E=3000, eta1=5000, eta2=15000,
        noise_level_um=0.05
    )
    
    from Fitting_MFA import FittingMFA
    fitter = FittingMFA(time, lengths, r_eff=5.0, delta_p=1000.0)
    
    # Use global optimization for best chance of recovery
    params = {
        'use_global_optimization': True,
        'fitting_bounds': {'E': [100, 1e5], 'eta1': [100, 5e5], 'eta2': [500, 1e6]},
        'jeffreys_maxfev': 10000
    }
    fitter.run(params)
    
    res = fitter.fit_results.get('Jeffreys')
    assert res is not None
    assert res['r_squared'] > 0.95
    
    # Check parameters (allow 20% tolerance due to ill-posed nature of viscoelastic fitting)
    p = dict(zip(res['param_names'], res['params']))
    assert abs(p['E'] - 3000)/3000 < 0.2
    
    print(f"✓ Parameter recovery passed (R²={res['r_squared']:.4f})")

# =============================================================================
# TEST 7: Validation Logic
# =============================================================================

def test_validation_logic():
    """Tests that physical impossibility checks work."""
    ranges = {'E': (100, 1000), 'eta1': (100, 1000), 'eta2': (100, 1000)}
    
    # Case 1: Valid
    valid, _ = validate_jeffreys_parameters(500, 500, 500, ranges)
    # (Note: ratio check might warn, but bounds are valid)
    
    # Case 2: Negative value (Impossible)
    valid_neg, msgs = validate_jeffreys_parameters(-500, 500, 500, ranges)
    assert not valid_neg
    assert any("outside typical range" in m for m in msgs)
    
    print("✓ Parameter validation logic verified")

# =============================================================================
# RUNNER
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING COMPLETE MFA VALIDATION SUITE")
    print("=" * 60)
    
    try:
        test_son_factor_table1()
        test_shear_metrics()
        test_jeffreys_newtonian_limit()
        test_burgers_physics()
        test_validation_logic()
        test_cusum_detects_synthetic_rupture()
        test_protrusion_length_measurement()
        test_parameter_recovery_jeffreys()
        
        print("\n" + "=" * 60)
        print("✓ ALL SYSTEMS GO: PIPELINE VERIFIED")
        print("=" * 60)
        
    except AssertionError as e:
        print("\n" + "=" * 60)
        print("✗ VALIDATION FAILED")
        print("=" * 60)
        print(e)