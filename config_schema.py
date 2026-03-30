# -*- coding: utf-8 -*-
"""
Pydantic models for configuration file validation.

ROLE IN PIPELINE:
This module enforces strict type checking on the 'config.yaml' file.
It ensures that the pipeline never starts with invalid parameters.
"""
import os
import math
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional, Any

from pydantic import BaseModel, Field, validator, ValidationError
from pydantic import ConfigDict
from pydantic import field_validator

logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS FOR AUTOMATIC f* CALCULATION  
# =============================================================================

def calculate_recommended_fstar(width_um: float, height_um: float) -> Tuple[float, float, float]:
    """
    Calculates the exact f* (Son factor) using Equation 20 from Son (2007).
    
    The Son factor corrects for the additional hydraulic resistance in rectangular
    channels compared to circular ones.
    
    Formula:
        f*(x) = [ (1 + 1/x)^2 * (1 - (192 / pi^5 * x) * SUM) ]^-1
        where x is the aspect ratio (H/W, always <= 1.0).
        
    Args:
        width_um: Channel width in micrometers
        height_um: Channel height in micrometers
        
    Returns:
        Tuple of (calculated_fstar, min_valid, max_valid)
        
    References:
        - Y. Son, Polymer 48 (2007) 632-637, Eq. 20 and Table 1.
    """
    # 1. Determine aspect ratio 'x' as defined by Son (H/W where H <= W)
    #    Range is 0.0 (Infinite Slit) to 1.0 (Square)
    dim_min = min(width_um, height_um)
    dim_max = max(width_um, height_um)
    x = dim_min / dim_max  
    
    # 2. Compute the infinite sum term (converges very rapidly due to 1/i^5)
    #    Son Eq 20: Sum_{i=1,3,5...} [ tanh(pi*i*x/2) / i^5 ]
    sum_term = 0.0
    # 11 terms (up to i=21) provides precision far beyond float capabilities
    for n in range(1, 22, 2):  
        term = math.tanh((math.pi * n * x) / 2) / (n ** 5)
        sum_term += term
        
    # 3. Calculate f*
    #    Son Eq 20: f* = [ (1 + 1/x)^2 * (1 - (192 / pi^5 * x) * SUM) ]^-1
    pi_5 = math.pi ** 5
    bracket_1 = (1 + 1.0/x) ** 2
    bracket_2 = 1 - (192 / (pi_5 * x)) * sum_term
    
    fstar = 1.0 / (bracket_1 * bracket_2)
    
    # Define a generic validity range for warnings (e.g. +/- 5%)
    min_val = fstar * 0.95
    max_val = fstar * 1.05
    
    logger.debug(
        f"Son Factor Calculation: H/W={x:.3f} (x) -> f*={fstar:.4f}"
    )
    
    return fstar, min_val, max_val


# =============================================================================
# 1. USER-FACING SETTINGS
# =============================================================================

class PathsConfig(BaseModel):
    """Configuration for input/output paths."""
    data_folder: Path
    experiment_id: str = Field(..., min_length=1)
    
    @field_validator('data_folder')
    @classmethod
    def folder_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Data folder does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Data folder path is not a directory: {v}")
        return v

class ExperimentConfig(BaseModel):
    """Core experimental parameters."""
    scale_factor: float = Field(gt=0.0, lt=10.0, description="μm/pixel")
    constant_pressure: float = Field(gt=0.0, lt=50000.0, description="Applied pressure in Pa")
    frame_interval: float = Field(gt=0.0, lt=10.0, description="Time between frames in seconds")
    max_traps: int = Field(gt=0, lt=100)

class ModelConfig(BaseModel):
    """Parameters for viscoelastic model fitting."""
    perform_fitting: bool = True
    # -------------------------
    channel_width_um: float = Field(gt=0.0, lt=100.0)
    channel_height_um: float = Field(gt=0.0, lt=100.0)
    channel_length_um: float = Field(gt=0.0, lt=1000.0, description="Length of the narrow pipette channel")
    
    fluid_viscosity_pa_s: float = Field(default=0.001, gt=0.0, lt=1.0, description="Fluid viscosity (Pa.s)")
    
    fstar: Optional[Union[float, str]] = Field(
        default="auto",
        description="Son's shape factor (0.1-1.0) or 'auto' for automatic calculation based on aspect ratio"
    )
    
    @field_validator('fstar', mode='before')
    @classmethod
    def validate_and_set_fstar(cls, v, info):
        """
        Validate fstar and auto-calculate if set to 'auto'.
        Also warns if manual fstar is outside recommended range.
        """
        values = info.data
        # Get channel dimensions
        width = values.get('channel_width_um')
        height = values.get('channel_height_um')
        
        if width is None or height is None:
            # Can't calculate without dimensions
            if v == "auto" or v is None:
                raise ValueError("Cannot auto-calculate fstar: channel dimensions not provided")
            return v
        
        # Calculate recommended fstar using Son's equation
        recommended, min_rec, max_rec = calculate_recommended_fstar(width, height)
        aspect_ratio = max(width, height) / min(width, height)
        
        # Handle 'auto' mode
        if v == "auto" or v is None:
            logger.info(
                f"Auto-calculated f* (Son Factor) for aspect ratio {aspect_ratio:.2f}:1 → {recommended:.4f}"
            )
            return recommended
        
        # Manual fstar provided - validate it
        if not isinstance(v, (int, float)):
            raise ValueError(f"fstar must be a number or 'auto', got: {v}")
        
        fstar_float = float(v)
        
        # Check hard limits
        if not (0.1 <= fstar_float <= 1.0):
            raise ValueError(f"fstar must be between 0.1 and 1.0, got: {fstar_float}")
        
        # Check against recommended range and warn if outside
        if fstar_float < min_rec or fstar_float > max_rec:
            logger.warning(
                f"\n{'='*70}\n"
                f"Son factor (f*) WARNING:\n"
                f"  Your f* = {fstar_float:.2f} is outside the calculated range\n"
                f"  for aspect ratio {aspect_ratio:.2f}:1 based on Son (2007).\n"
                f"  \n"
                f"  Calculated (Eq. 20): f* = {recommended:.4f}\n"
                f"  \n"
                f"  Options:\n"
                f"  1. Use 'fstar: auto' to auto-calculate (recommended)\n"
                f"  2. Keep your value if based on specific calibration\n"
                f"{'='*70}\n"
            )
        
        return fstar_float
    
    @field_validator('channel_height_um')
    @classmethod
    def validate_aspect_ratio(cls, v: float, info) -> float:
        """Check if channel aspect ratio is within reasonable bounds."""
        values = info.data
        if 'channel_width_um' in values:
            width = values['channel_width_um']
            aspect_ratio = max(width, v) / min(width, v)
            if aspect_ratio > 10:
                logger.warning(
                    f"Unusual channel aspect ratio detected: {aspect_ratio:.1f}:1 "
                    f"(Width={width:.1f}µm, Height={v:.1f}µm). "
                    f"Verify your channel dimensions are correct."
                )
        return v

class RuptureDetectionConfig(BaseModel):
    """Parameters for detecting membrane rupture and filtering data."""
    enable: bool = True
    intensity_threshold_std: float = 3.0
    rupture_offset_from_tip_px: int = 10
    rupture_window_width_px: int = 15
    
    entry_protrusion_threshold_um: float = Field(default=0.5, ge=0.0, description="Length (um) to mark cell entry")
    exit_protrusion_threshold_um: float = Field(default=0.5, ge=0.0, description="Length drop (um) to mark cell exit")
    exit_drop_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="Fractional drop to mark cell exit")

    doa_retention_ratio: float = Field(default=0.80, ge=0.0, le=10.0, description="Signal stays above 80% of empty trap")
    doa_solidity_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Mask is fragmented")

    max_baseline_sigma: float = Field(default=1.0, gt=0.0, description="Cap on baseline variance to prevent blinding")
    min_cusum_baseline_frames: int = Field(default=5, ge=3, description="Min frames required to calculate CUSUM baseline")
    min_intensity_noise_floor: float = Field(default=0.2, gt=0.0, description="Minimum sigma to prevent CUSUM division by zero")
    min_drift_tolerance: float = Field(default=0.2, gt=0.0, description="Minimum drift K (intensity units)")
    min_cusum_threshold: float = Field(default=2.0, gt=0.0, description="Minimum threshold H (intensity units)")

    exclude_early_fraction: float = Field(ge=0.0, lt=0.2)
    
    enable_outlier_rejection: bool = True
    outlier_rejection_window: int = Field(gt=3, lt=51)
    outlier_rejection_std_dev: float = Field(gt=0.0, lt=10.0)
    enable_smoothing: bool = Field(default=True, description="Apply rolling median smoothing to raw length data")
    
    cusum_settling_buffer: int = Field(default=3, ge=0, description="Frames to skip after entry before monitoring rupture")
    cusum_baseline_lag: int = Field(default=0, ge=0, description="Frames to gap between baseline and test window")
    cusum_baseline_len: int = Field(default=5, ge=3, description="Number of frames to establish baseline noise")
    cusum_sensitivity_sigma: float = Field(default=0.7413, gt=0.0, description="Multiplier for IQR to determine noise sigma")
    
    cusum_drift_tolerance_factor: float = Field(default=0.5, gt=0.0, le=2.0, description="Drift tolerance as multiple of sigma (k parameter)")
    cusum_threshold_factor: float = Field(default=10.0, gt=1.0, le=50.0, description="Detection threshold as multiple of sigma (h parameter)")
    
    enable_spike_check: bool = Field(default=True)
    spike_sigma_threshold: float = Field(default=6.0, gt=0.0)
    enable_step_check: bool = Field(default=True)
    step_sigma_threshold: float = Field(default=6.0, gt=0.0)
    
    sudden_jump_threshold: float = Field(default=1.0, gt=0.0, description="Intensity difference to trigger acute rupture")
    
    # Pulse context detection
    pulse_context_pre_window: int = Field(default=5, ge=1)
    pulse_context_post_window: int = Field(default=5, ge=1)
    pulse_context_peak_window: int = Field(default=2, ge=1)
    pulse_context_fold_threshold: float = Field(default=1.3, gt=1.0)
    pulse_context_abs_threshold: float = Field(default=0.5, gt=0.0)
    
    # Exit guards
    pulse_exit_blanking_frames: int = Field(default=5, ge=0)
    
    # Anchored drift check
    anchored_baseline_frames: int = Field(
        default=5, ge=2,
        description="Fixed early frames used as baseline for anchored drift detector"
    )
    anchored_fold_threshold: float = Field(
        default=1.5, gt=1.0,
        description="Later window must be this multiple of anchored baseline to flag rupture"
    )
    anchored_abs_threshold: float = Field(
        default=1.0, gt=0.0,
        description="Minimum absolute intensity rise required alongside fold threshold"
    )
    anchored_scan_window: int = Field(
        default=5, ge=2,
        description="Size of sliding window used by anchored drift detector"
    )
    
    @field_validator('outlier_rejection_window')
    @classmethod
    def window_must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError(f"Outlier rejection window size must be odd, got {v}")
        return v

class GuvSettings(BaseModel):
    """Settings that apply specifically to GUV (Giant Unilamellar Vesicle) experiments.

    Set enable=True to activate GUV mode. Segmentation uses edge-gradient
    (Canny) detection rather than intensity thresholding, which produces
    consistent masks across frames even when absolute pixel intensity drifts.
    """
    enable: bool = Field(default=False, description="Master switch for GUV mode.")

    selection_frame_index: int = Field(
        default=2,
        ge=0,
        description=(
            "Frame shown during the interactive threshold-selection GUI step. "
            "Use an early frame (e.g. 2) where the membrane is still intact "
            "and spherical."
        )
    )

    canny_sigma: float = Field(
        default=0.4,
        ge=0.05,
        le=1.0,
        description=(
            "Width of the auto-Canny band around the per-frame median intensity. "
            "low = (1 - sigma) * median, high = (1 + sigma) * median. "
            "Increase (0.5-0.6) for dim membranes; decrease (0.2-0.3) to suppress debris edges."
        )
    )

class WorkflowConfig(BaseModel):
    """Settings that control the pipeline's execution flow."""
    verify_traps_interactively: bool = True
    create_kymographs: bool = True
    window_scale_factor: float = Field(default=1.0, gt=0.0, le=5.0, description="Scaling factor for interactive window size on high-DPI screens")
    max_cache_size: int = Field(gt=10, lt=500)
    trap_spacing_factor: float = Field(gt=1.0, lt=3.0)
    debug_mode: bool = False

    interactive_window_width: int = 1000
    interactive_window_height: int = 800
    default_roi_width_um: float = 60.0
    default_roi_height_um: float = 25.0

    n_workers: int = Field(ge=1, le=(os.cpu_count() or 1))
    selection_frame_fraction: float = Field(ge=0.0, le=1.0)
    

# =============================================================================
# 2. DEVELOPER-LEVEL ALGORITHM & PLOTTING SETTINGS
# =============================================================================

class ImageProcessingConstants(BaseModel):
    """Advanced settings for image processing algorithms."""
    wall_clip_margin: float = Field(default=0.40, ge=0.0, lt=0.5)
    clahe_clip_limit: float = Field(default=2.0, ge=1.0, le=5.0)
    clahe_tile_grid_size: Tuple[int, int] = Field(default=(8, 8))
    gaussian_kernel_size: Tuple[int, int] = Field(default=(3, 3))
    min_area_threshold: int = Field(default=100, ge=1, description="Minimum blob area to track [pixels²]")
    small_object_threshold: int = Field(default=50, ge=1, description="Remove objects smaller than this [pixels²]")

    @field_validator('clahe_tile_grid_size')
    @classmethod
    def grid_size_valid(cls, v) -> tuple:
        if not (2 <= v[0] <= 16 and 2 <= v[1] <= 16):
            raise ValueError(f"Grid size components must be between 2 and 16, got {v}")
        return v

class KymographConstants(BaseModel):
    """Advanced settings for kymograph generation."""
    default_kymograph_width_px: int = Field(default=100)
    default_line_width_px: int = Field(default=10)

class PlottingConstants(BaseModel):
    """Default settings for generating Matplotlib figures."""
    default_dpi: int = Field(default=300, gt=70)
    figure_size_large: Tuple[float, float] = Field(default=(18.0, 12.0))
    figure_size_medium: Tuple[float, float] = Field(default=(10.0, 7.0))

class FittingParameters(BaseModel):
    """Advanced settings for the optimization."""
    use_global_optimization: bool = Field(default=False)
    
    jeffreys_maxfev: int = Field(default=50000)
    empirical_maxfev: int = Field(default=5000)
    default_initial_guess: Tuple[float, float, float] = Field(default=(3000.0, 5000.0, 15000.0))
    
    fitting_bounds: Dict[str, Tuple[float, float]] = Field(default={
        "E": (100.0, 100000.0),
        "eta1": (100.0, 500000.0),
        "eta2": (500.0, 1000000.0)
    })
    
    clipping_bounds_guess: Dict[str, Tuple[float, float]] = Field(default={
        "E": (500.0, 25000.0),
        "eta1": (1000.0, 50000.0),
        "eta2": (5000.0, 100000.0)
    })

class ValidationParameters(BaseModel):
    """Physical plausibility ranges for validating fitted results."""
    plausible_ranges: Dict[str, Tuple[float, float]] = Field(default={
        "E": (100.0, 50000.0),
        "eta1": (500.0, 100000.0),
        "eta2": (1000.0, 200000.0)
    })

# =============================================================================
# 3. TOP-LEVEL CONFIGURATION MODEL
# =============================================================================

class DyeUptakeConfig(BaseModel):
    """Configuration for electroporation dye uptake analysis."""
    enable: bool = False
    has_pulse: bool = Field(default=True, description="Set to False for control experiments without electroporation.")
    
    # Filename patterns to distinguish channels (e.g., "TRITC" vs "FITC")
    membrane_channel_pattern: str = Field(default="C1", description="Substring to identify membrane images")
    dye_channel_pattern: str = Field(default="C2", description="Substring to identify dye images")
    
    # Analysis parameters
    pulse_frame: float = Field(default=10.0, ge=0.1, le=1000.0, description="Frame number where pulse is applied (1-based index)")
    pulse_index: int = Field(default=9, description="0-based index calculated automatically")
    baseline_frames: int = Field(default=5, ge=1, description="Number of pre-pulse frames to average for baseline")
    
    @field_validator('pulse_index', mode='before')
    @classmethod
    def calculate_pulse_index(cls, v, info) -> int:
        return max(0, int(info.data.get('pulse_frame', 10.0)) - 1)
    
class ActinConfig(BaseModel):
    """Configuration for actin distribution analysis."""
    enable: bool = False
    channel_pattern: str = Field(default="C3", description="Substring to identify actin images")
    cortex_thickness_px: int = Field(default=3, ge=1, description="Cortex shell depth in pixels (3px ≈ 1.9µm at 60x)")
    cortex_cv_threshold: float = Field(default=0.4, gt=0.0, lt=2.0, description="CV threshold: below = uniform cortex, above = patchy")
    pre_pulse_window_frames: int = Field(default=5, ge=1, description="Frames before pulse for remodelling score")
    post_pulse_window_frames: int = Field(default=10, ge=1, description="Frames after pulse for remodelling score")

class MFAConfig(BaseModel):
    """The master schema that combines all sub-configurations."""
    
    paths: PathsConfig
    experiment_parameters: ExperimentConfig
    guv_settings: GuvSettings = Field(default_factory=GuvSettings)
    model_parameters: ModelConfig
    rupture_detection: RuptureDetectionConfig
    workflow_settings: WorkflowConfig
    dye_uptake_parameters: DyeUptakeConfig = Field(default_factory=DyeUptakeConfig)
    actin_parameters: ActinConfig = Field(default_factory=ActinConfig)
    
    image_processing: ImageProcessingConstants = Field(default_factory=ImageProcessingConstants)
    kymograph_parameters: KymographConstants = Field(default_factory=KymographConstants)
    plotting_parameters: PlottingConstants = Field(default_factory=PlottingConstants)
    fitting_parameters: FittingParameters = Field(default_factory=FittingParameters)
    validation_parameters: ValidationParameters = Field(default_factory=ValidationParameters)

    model_config = ConfigDict(extra='forbid')
        
# =============================================================================
# 4. VALIDATION FUNCTION
# =============================================================================

def validate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates the raw dictionary against the MFAConfig schema.
    Returns the validated configuration as a dictionary (with defaults applied).
    """
    try:
        # Create Pydantic model (performs validation)
        config_obj = MFAConfig(**config_dict)
        
        # Convert back to dictionary for compatibility with existing scripts
        # Attempt .model_dump() (Pydantic v2) or fallback to .dict() (Pydantic v1)
        if hasattr(config_obj, 'model_dump'):
            return config_obj.model_dump()
        else:
            return config_obj.dict()
            
    except ValidationError as e:
        logger.error(f"Configuration Validation Failed!\n{e}")
        raise e