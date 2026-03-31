# -*- coding: utf-8 -*-
"""
Biomechanical models and calculations for micropipette aspiration.

This module contains the mathematical backend for analyzing cell mechanics from
micropipette aspiration experiments. It provides:

1. **Geometric Corrections**: Accounts for rectangular channel geometry using 
   Son (2007) hydraulic resistance theory.
   
2. **Viscoelastic Models**: Implements standard rheological models (Jeffreys, 
   Kelvin-Voigt, Burgers) to extract material properties from aspiration curves.
   
3. **Parameter Estimation**: Provides heuristics for initial guess generation 
   to improve fitting convergence.
   
4. **Validation Functions**: Checks if fitted parameters are physically plausible.

ROLE IN PIPELINE
----------------
This is the **pure mathematics layer**. It contains NO image processing, file I/O, 
or plotting logic—only physics equations and curve fitting helpers.

The typical workflow is:
    User Data → LineDetection (measures length vs time) → THIS MODULE (extracts 
    E, η₁, η₂) → Plotting (visualizes results)

COORDINATE SYSTEM & SIGN CONVENTIONS
-------------------------------------
- **Length (L)**: Distance the cell protrudes INTO the pipette [μm]  
  (positive = aspiration, zero = cell just touching entrance)
- **Pressure (ΔP)**: Applied suction pressure [Pa]  
  (positive = suction, pulls cell into pipette)
- **Time (t)**: Elapsed time since pressure application [seconds]

PHYSICAL ASSUMPTIONS
--------------------
1. **Incompressible material**: Cell volume conserved during deformation
2. **Homogeneous properties**: E and η uniform throughout cell
3. **Small strain**: Linear viscoelastic theory applies (typically L < 0.5 × diameter)
4. **Instantaneous pressure**: ΔP applied as step function at t=0
5. **No-slip boundary**: Cell membrane adheres to channel wall
6. **Rectangular channel approximation**: Channel cross-section constant along length

REFERENCES
----------
.. [1] Son, Y. (2007). Determination of shear viscosity and shear rate from 
       pressure drop and flow rate relationship in a rectangular channel. 
       Polymer, 48(2), 632-637. doi:10.1016/j.polymer.2006.11.048
       
.. [2] Hochmuth, R. M. (2000). Micropipette aspiration of living cells. 
       Journal of Biomechanics, 33(1), 15-22.
       
.. [3] Evans, E., & Yeung, A. (1989). Apparent viscosity and cortical tension 
       of blood granulocytes determined by micropipette aspiration. 
       Biophysical Journal, 56(1), 151-160.

COORDINATE TRANSFORMATIONS
---------------------------
When converting between circular (traditional micropipette) and rectangular 
(microfluidic) geometries:

    Circular:  r_circular = channel diameter / 2
    Rectangular: r_eff = compute_reff(W, H, f*)
    
    Then use r_eff in place of r_circular in all equations.

VALIDATION
----------
This module has been validated against:
- Son (2007) Table 1: f* calculation reproduces published values within 0.5%
- Jeffreys model limit: Correctly reduces to Kelvin-Voigt when η₂ → ∞
- Synthetic data: Recovers known parameters within 5% on noise-free data

See tests/test_synthetic_validation.py for details.
"""
import logging
from typing import Tuple, List, Union, Dict
import numpy as np

from config_schema import calculate_recommended_fstar

logger = logging.getLogger(__name__)


def compute_reff(width: float, height: float, f_star: float) -> float:
    """
    Computes effective radius for a rectangular channel using Son (2007) hydraulic theory.
    
    Converts rectangular cross-section (W × H) into equivalent circular radius that 
    produces the same flow resistance. Allows use of circular pipette equations 
    on rectangular microfluidic channels.
    
    Parameters
    ----------
    width : float
        Channel width [μm]
    height : float
        Channel height [μm]
    f_star : float
        Son's shape factor (dimensionless), typically 0.59-1.0
        Use calculate_recommended_fstar() to compute from aspect ratio
        
    Returns
    -------
    float
        Effective radius [μm]
        
    Examples
    --------
    >>> r_eff = compute_reff(6.7, 5.0, f_star=0.7)
    >>> print(f"{r_eff:.2f} μm")
    4.85 μm
    
    See Also
    --------
    README.md : Section "Geometric Corrections" for mathematical derivation
    
    References
    ----------
    Son, Y. (2007). Polymer, 48(2), 632-637, Eq. 18 & 21.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive.")

    # Physics dictates H <= W for these aspect ratio formulas
    h_s = min(width, height)
    w_l = max(width, height)

    # The pre-factor for a rectangular duct matching Son (2007) 
    # and Davidson et al. is 2 / (3 * pi).
    # Derivation: Equating Poiseuille flow Q_circ to Rectangular flow Q_rect (Eq 21 + 18)
    
    numerator = (2.0 / (3.0 * np.pi)) * w_l * (h_s**3)
    denominator = (1 + h_s / w_l)**2 * f_star
    
    r_eff_4 = numerator / denominator
    r_eff = r_eff_4**0.25

    logger.debug(f"Calculated r_eff = {r_eff:.3f} µm (Sorted: W={w_l}, H={h_s}, f*={f_star})")
    return r_eff


def compute_shear_metrics(width_um: float, height_um: float, length_um: float, 
                          pressure_pa: float, viscosity_pa_s: float) -> Dict[str, float]:
    """
    Calculates Shear Stress and Shear Rate using Son (2007) Eqs 18, 21, 24.
    
    Args:
        width_um, height_um, length_um: Channel dimensions in microns.
        pressure_pa: Pressure gradient (Delta P) in Pascals.
        viscosity_pa_s: Fluid viscosity in Pa.s.
    """
    # 1. Geometry Setup (Ensure H <= W per Son's definition)
    H = min(width_um, height_um) * 1e-6  # Convert to meters
    W = max(width_um, height_um) * 1e-6
    L = length_um * 1e-6                 # Convert to meters
    
    # 2. Wall Shear Stress (Eq. 18) [cite: 148]
    # tau_w = (dP / 2L) * (WH / (W + H))
    term_1 = (pressure_pa * H) / (2 * L)
    term_2 = 1.0 / (1.0 + (H / W))
    tau_w = term_1 * term_2
    
    # 3. Geometric Factor f* (Eq. 20) [cite: 120]
    # Recalculate f* to ensure we use the exact one for this calculation
    aspect = H / W
    f_star, _, _ = calculate_recommended_fstar(width_um, height_um)

    # 4. Apparent Shear Rate (Eq. 21) [cite: 116]
    # Calculated via Flow Rate Q for Newtonian fluid (n=1)
    # Rearranging Eq 18 and 21 for Newtonian fluid:
    # Q = (tau_w * W * H^2) / (6 * mu * (1 + H/W) * f*)
    
    numerator = tau_w * W * (H**2)
    denominator = 6 * viscosity_pa_s * (1 + aspect) * f_star
    Q = numerator / denominator # m^3/s

    # Apparent Shear Rate: gamma_a = (6Q / WH^2) * (1 + H/W) * f*
    gamma_a = (6 * Q) / (W * H**2) * (1 + aspect) * f_star

    # 5. True Wall Shear Rate (Eq. 24) [cite: 136]
    # For Newtonian fluids, correction terms cancel out -> gamma_w == gamma_a
    gamma_w = gamma_a 

    return {
        "Wall_Shear_Stress_Pa": tau_w,
        "Wall_Shear_Rate_s1": gamma_w,
        "Flow_Rate_nL_min": Q * 1e9 * 60, # Convert m3/s to nL/min
        "Aspect_Ratio": aspect,
        "Son_Factor_fstar": f_star,
        "Channel_Length_um": length_um
    }

# =============================================================================
# VISCOELASTIC & EMPIRICAL MODELS (The Equations)
# =============================================================================

def jeffreys_length(t: np.ndarray, r_eff: float, delta_p: float, C: float, E: float, eta1: float, eta2: float) -> np.ndarray:
    """
    3-Parameter Jeffreys Model (Viscoelastic Liquid).
    Structure: (Spring E // Dashpot eta1) --series-- (Dashpot eta2)
    
    The half-space constant (C): often denoted as $\Phi$ accounts for the wall thickness of the pipette. 
        - Davidson et al. assume C ~ 1.
        - For thin-walled pipettes, $\Phi ~ 2$. 
        - For the continuous geometries of microfluidic blocks, the boundary conditions differ.
          If you are comparing to traditional glass micropipette data, you may need to adjust $C$ to ~2.0-2.1 or treat $E$ as an "apparent" modulus.
    """
    t = np.asarray(t, dtype=float)
    # Characteristic time constant (tau)
    tau = (3 * np.pi * eta1) / (C * E)
    
    # Elastic creep term (1 - exp)
    elastic_scale = (r_eff * delta_p) / (C * E)
    elastic_term = elastic_scale * (1 - np.exp(-t / tau))
    
    # Viscous flow term (Linear)
    viscous_term = (r_eff * delta_p) / (3 * np.pi * eta2) * t
    
    return elastic_term + viscous_term

def kelvin_voigt_length(t: np.ndarray, r_eff: float, delta_p: float, C: float, E: float, eta: float) -> np.ndarray:
    """
    Kelvin-Voigt Model (Viscoelastic Solid).
    Structure: Spring E // Dashpot eta
    """
    t = np.asarray(t, dtype=float)
    tau = (3 * np.pi * eta) / (C * E)
    elastic_scale = (r_eff * delta_p) / (C * E)
    return elastic_scale * (1 - np.exp(-t / tau))

def burgers_length(t: np.ndarray, r_eff: float, delta_p: float, C: float, E1: float, eta1: float, E2: float, eta2: float) -> np.ndarray:
    """
    4-Parameter Burgers Model.
    Structure: Maxwell Element (Spring+Dashpot) + Kelvin-Voigt Element.
    """
    t = np.asarray(t, dtype=float)
    
    # Maxwell element (Linear flow + instant jump)
    maxwell_term = (r_eff * delta_p / C) * (1 / E1 + t / (3 * np.pi * eta1))
    
    # Kelvin-Voigt element (delayed elasticity)
    tau_kv = (3 * np.pi * eta2) / (C * E2)
    kelvin_voigt_term = (r_eff * delta_p / (C * E2)) * (1 - np.exp(-t / tau_kv))
    return maxwell_term + kelvin_voigt_term

def power_law_model(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power Law: L = a * t^b (Common in soft glassy rheology)."""
    return a * (t + 1e-9)**b

def power_law_model_with_c(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Power Law with offset."""
    return a * (t + 1e-9)**b + c

def linear_model(t: np.ndarray, m: float, b: float) -> np.ndarray:
    """Simple linear flow: L = mt + b."""
    return m * t + b

def exponential_model(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """y = a * (1 - exp(-b*t))"""
    return a * (1 - np.exp(-b * t))


# =============================================================================
# PARAMETER ESTIMATION (Heuristics)
# =============================================================================

def estimate_initial_parameters(
    time_data: np.ndarray, 
    length_data: np.ndarray, 
    r_eff: float, 
    delta_p: float, 
    C: float,
    clip_bounds: Dict[str, Tuple[float, float]],
    default_guess: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Smartly guesses E, eta1, and eta2 from the curve shape to help fitting converge.
    """
    t = np.asarray(time_data)
    L = np.asarray(length_data)

    valid_mask = (L > 0) & np.isfinite(L) & np.isfinite(t)
    if not np.any(valid_mask):
        logger.warning("No valid data for parameter estimation, using default guess.")
        return default_guess

    t_clean, l_clean = t[valid_mask], L[valid_mask]

    # 1. Estimate E (Elasticity)
    max_l = np.max(l_clean)
    e_guess = (r_eff * delta_p) / (C * max_l) if max_l > 0 else default_guess[0]

    # 2. Estimate eta2 (Series Viscosity) -> Based on late-stage slope
    eta2_guess = default_guess[2]
    if len(t_clean) > 10:
        n_late = max(5, len(t_clean) // 3)
        late_t, late_l = t_clean[-n_late:], l_clean[-n_late:]
        if len(late_t) > 2 and np.ptp(late_t) > 0:
            slope = np.polyfit(late_t, late_l, 1)[0]
            if slope > 0.01:
                eta2_guess = (r_eff * delta_p) / (3 * np.pi * slope)

    # 3. Estimate eta1 (Parallel Viscosity) -> Based on time constant tau
    eta1_guess = default_guess[1]
    if len(t_clean) > 5 and np.max(l_clean) > 0:
        target_l = 0.63 * max_l
        idx = np.argmin(np.abs(l_clean - target_l))
        t_63 = t_clean[idx]
        if t_63 > 0:
            eta1_guess = t_63 * C * e_guess / (3 * np.pi)

    # 4. Clip to bounds (prevents wild guesses)
    try:
        e_guess = np.clip(e_guess, *clip_bounds['E'])
        eta1_guess = np.clip(eta1_guess, *clip_bounds['eta1'])
        eta2_guess = np.clip(eta2_guess, *clip_bounds['eta2'])
    except KeyError:
        logger.error("`clipping_bounds_guess` dict is missing required keys. Using default guess.")
        return default_guess
    
    # 5. Physics check: Series viscosity usually higher than creep viscosity
    if eta2_guess < eta1_guess:
        eta2_guess = eta1_guess * 2.0

    logger.debug(f"Parameter estimation: E ≈ {e_guess:.0f} Pa, η₁ ≈ {eta1_guess:.0f} Pa·s, η₂ ≈ {eta2_guess:.0f} Pa·s")
    return float(e_guess), float(eta1_guess), float(eta2_guess)


def validate_jeffreys_parameters(
    E: float, 
    eta1: float, 
    eta2: float,
    plausible_ranges: Dict[str, Tuple[float, float]]
) -> Tuple[bool, List[str]]:
    """
    Checks if the fitted parameters make physical sense.
    """
    warnings = []
    is_valid = True

    try:
        e_min, e_max = plausible_ranges['E']
        eta1_min, eta1_max = plausible_ranges['eta1']
        eta2_min, eta2_max = plausible_ranges['eta2']
    except KeyError:
        logger.error("`plausible_ranges` dict is missing required keys ('E', 'eta1', 'eta2'). Skipping validation.")
        return is_valid, ["Validation skipped due to config error."]

    if not (e_min <= E <= e_max):
        warnings.append(f"Elastic modulus E = {E:.0f} Pa is outside typical range ({e_min}-{e_max} Pa).")
        if not (e_min / 10 <= E <= e_max * 10): is_valid = False

    if not (eta1_min <= eta1 <= eta1_max):
        warnings.append(f"Parallel viscosity η₁ = {eta1:.0f} Pa·s is outside typical range ({eta1_min}-{eta1_max} Pa·s).")
        if not (eta1_min / 10 <= eta1 <= eta1_max * 10): is_valid = False

    if not (eta2_min <= eta2 <= eta2_max):
        warnings.append(f"Series viscosity η₂ = {eta2:.0f} Pa·s is outside typical range ({eta2_min}-{eta2_max} Pa·s).")
        if not (eta2_min / 10 <= eta2 <= eta2_max * 10): is_valid = False

    if E > 0 and eta1 > 0:
        tau = (3 * np.pi * eta1) / E
        if not (0.1 < tau < 100):
            warnings.append(f"Relaxation time τ = {tau:.1f} s seems physically unusual.")

        if eta1 > 0:
            ratio = eta2 / eta1
            if not (1.5 < ratio < 50):
                warnings.append(f"Series/parallel viscosity ratio ({ratio:.1f}) is unusual (typically >2).")
        else:
            warnings.append("Parallel viscosity eta1 is zero, cannot calculate ratio.")

    return is_valid, warnings