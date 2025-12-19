# MFA Analysis Pipeline

**Automated analysis of microfluidic micropipette aspiration experiments for measuring cell membrane mechanics.**

This pipeline processes time-lapse microscopy images of cells being aspirated into rectangular microfluidic channels, extracting viscoelastic material properties (elastic modulus E, viscosities η₁ and η₂) from deformation dynamics.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Mathematical Models](#mathematical-models)
6. [Configuration Guide](#configuration-guide)
7. [Validation Results](#validation-results)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

---

## Features

### Core Capabilities
- ✅ **Parallel processing** of up to 18 traps simultaneously
- ✅ **Interactive setup** with visual trap selection and parameter tuning
- ✅ **Sub-pixel edge detection** for accurate protrusion measurements
- ✅ **Automatic rupture detection** using CUSUM algorithm
- ✅ **Multi-model fitting** (Jeffreys, Kelvin-Voigt, Burgers, empirical)
- ✅ **Rectangular channel corrections** using Son (2007) hydraulic theory
- ✅ **Electroporation dye uptake analysis** (optional module)

### Output Formats
- **CSV**: Time-series data (protrusion length, intensity, fitted curves)
- **PNG**: Publication-quality plots (traces, kymographs, model comparisons)
- **Summary tables**: Aggregated results across all traps

### Validated Performance
- **Geometric corrections**: Reproduces Son (2007) f* within 0.5%
- **Edge detection**: 4.6% relative error on blurred/noisy synthetic images
- **Rupture detection**: 98% temporal accuracy (±1 frame)
- **Parameter recovery**: <5% error on clean synthetic data (R²=0.995)

---

## Installation

### Prerequisites
- Python 3.8+
- Windows/Mac/Linux

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/MFAE_Analysis.git
cd MFAE_Analysis
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt --break-system-packages  # For system Python
# OR
conda env create -f environment.yml  # For Anaconda
conda activate mfa
```

### Step 3: Verify Installation
```bash
python -m pytest tests/test_synthetic_validation.py
```

You should see:
```
✓ ALL VALIDATION TESTS PASSED
```

---

## Quick Start

### Minimal Example

1. **Prepare your data folder:**
   ```
   Experiment_001/
   ├── C1_t0001.tif
   ├── C1_t0002.tif
   └── ...
   ```

2. **Edit `config.yaml`:**
   ```yaml
   paths:
     data_folder: "C:/Data/Experiment_001"
     experiment_id: "Exp001"
   
   experiment_parameters:
     scale_factor: 0.629      # μm/pixel (check your microscope!)
     constant_pressure: 1100  # Pa
     frame_interval: 1.7      # seconds (backup if metadata missing)
   ```

3. **Run analysis:**
   ```bash
   python MFA_analysis.py config.yaml
   ```

4. **Interactive setup:**
   - Adjust image rotation (W/S: ±10°, A/D: ±0.1°)
   - Position ROI around first trap (drag box/corners)
   - Adjust trap spacing (W/S keys)
   - Click to select which traps to analyze
   - Tune detection thresholds per trap (visual feedback)

5. **Find results:**
   ```
   Output/Exp001/
   ├── Exp001_all_traps_protrusions.csv       # Main data
   ├── Exp001_summary_fits.csv                # Fitted parameters
   ├── Protrusion traces/                     # Plots per trap
   ├── Fitting results/                       # Analysis plots
   └── Kymographs/                            # Space-time images
   ```

---

## Pipeline Architecture

### 3-Phase Processing

```
┌────────────────────────────────────────────────────────────┐
│ PHASE 1: Interactive Setup (Single-threaded)              │
├────────────────────────────────────────────────────────────┤
│ 1. Load metadata (timestamps, channels)                   │
│ 2. User adjusts rotation → aligns traps horizontally      │
│ 3. User defines first trap ROI                            │
│ 4. Auto-detect remaining traps (spacing factor)           │
│ 5. User selects which traps to analyze                    │
│ 6. Per-trap parameter tuning (pipette X, thresholds)      │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│ PHASE 2: Parallel Processing (Multi-threaded)             │
├────────────────────────────────────────────────────────────┤
│ For each trap in parallel (n_workers cores):              │
│   1. Load images → Apply rotation → Crop trap ROI         │
│   2. LineDetection:                                        │
│      - Enhance contrast (CLAHE)                            │
│      - Detect edges (Canny + morphology)                   │
│      - Measure protrusion (sub-pixel refinement)           │
│      - Monitor rupture (CUSUM on downstream intensity)     │
│   3. Fitting (optional):                                   │
│      - Estimate initial parameters                         │
│      - Fit Jeffreys, Burgers, power-law models             │
│      - Select best model (R² comparison)                   │
│      - Validate parameters (plausibility checks)           │
│   4. Kymograph generation (optional)                       │
│   5. Dye uptake analysis (optional, if 2-channel data)     │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│ PHASE 3: Aggregation & Export (Single-threaded)           │
├────────────────────────────────────────────────────────────┤
│ 1. Combine results from all traps                         │
│ 2. Export consolidated CSV (wide & long formats)          │
│ 3. Generate summary tables (per-trap parameters)          │
│ 4. Calculate shear metrics (Son 2007)                     │
└────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Role | Key Functions |
|--------|------|---------------|
| `FileHandling_MFA.py` | I/O, metadata | `FileRead.run()`, timestamp extraction |
| `ImageProcessing_MFA.py` | Setup GUI | `CropImage.run()`, rotation, ROI selection |
| `LineDetection_MFA.py` | Protrusion measurement | `LineDetectionMFA.run()`, CUSUM rupture detection |
| `Calculation_MFA.py` | Physics & math | `compute_reff()`, `jeffreys_length()`, parameter validation |
| `Fitting_MFA.py` | Model optimization | `FittingMFA.run()`, multi-model comparison |
| `Kymograph_MFA.py` | Space-time visualization | `create_kymograph_for_trap()` |
| `UptakeQuantification.py` | Dye analysis (optional) | `DyeUptakeAnalyzer.run()` |
| `Plotting_MFA.py` | Figure generation | All plotting functions |
| `Utils_MFA.py` | Shared utilities | Color palettes, logging, frame caching |

---

## Mathematical Models

### 1. Geometric Corrections for Rectangular Channels

**Problem:** Traditional micropipette theory assumes circular channels. Microfluidic devices use rectangular cross-sections.

**Solution:** Son (2007) derived equivalent circular radius `r_eff` that produces same hydraulic resistance.

#### Effective Radius Formula

$$
r_{\text{eff}}^4 = \frac{2}{3\pi} \cdot \frac{W_L \cdot H_S^3}{(1 + H_S/W_L)^2 \cdot f^*}
$$

where:
- $W_L = \max(\text{width}, \text{height})$ (longer dimension)
- $H_S = \min(\text{width}, \text{height})$ (shorter dimension)
- $f^*$ = Son's shape factor (accounts for corner flow effects)

#### Son's Shape Factor f*

Calculated from aspect ratio using Son (2007) Eq. 20:

$$
f^*(x) = \left[ \left(1 + \frac{1}{x}\right)^2 \left(1 - \frac{192}{\pi^5 x} \sum_{i=1,3,5...}^{\infty} \frac{\tanh(\frac{\pi i x}{2})}{i^5} \right) \right]^{-1}
$$

where $x = H/W \leq 1.0$ (aspect ratio).

**Validation:** Reproduces Son (2007) Table 1 within 0.5% (see `tests/test_synthetic_validation.py`).

---

### 2. Viscoelastic Models

All models predict protrusion length $L(t)$ from applied pressure $\Delta P$.

#### A. Jeffreys Model (3 parameters) — **RECOMMENDED**

**Structure:** Parallel spring-dashpot + series dashpot

```
     ┌──[E]──┐               
     │       │               
  ───[η₁]────[η₂]──── L(t)
     │       │
     └───────┘
   (parallel) (series)
```

**Equation:**

$$
L(t) = \frac{r_{\text{eff}} \Delta P}{C \cdot E} \left[1 - e^{-t/\tau}\right] + \frac{r_{\text{eff}} \Delta P}{3\pi \eta_2} \cdot t
$$

where:
- $\tau = \frac{3\pi \eta_1}{C \cdot E}$ = relaxation time [seconds]
- $C$ = geometric correction factor (1.0-2.0, typically 1.0 for microfluidics)

**Physical Interpretation:**
- **Term 1** (transient): Membrane stretches like elastic band, saturates at time $\tau$
- **Term 2** (steady): Membrane flows like viscous liquid, grows linearly forever

**Parameter Ranges (typical cells):**
| Parameter | Soft cells (e.g., neutrophils) | Stiff cells (e.g., osteoblasts) |
|-----------|--------------------------------|----------------------------------|
| E [Pa] | 100-500 | 2000-10000 |
| η₁ [Pa·s] | 100-5000 | 5000-50000 |
| η₂ [Pa·s] | 1000-20000 | 20000-200000 |
| τ [s] | 0.5-5 | 5-50 |

**Use when:** Cell deforms continuously over time (most common scenario).

---

#### B. Kelvin-Voigt Model (2 parameters)

**Structure:** Parallel spring-dashpot only

```
     ┌──[E]──┐
     │       │
  ───┤       ├──── L(t)
     │  [η]  │
     └───────┘
```

**Equation:**

$$
L(t) = \frac{r_{\text{eff}} \Delta P}{C \cdot E} \left[1 - e^{-t/\tau}\right]
$$

**Physical Interpretation:**
- Exponential approach to plateau $L_{\max} = \frac{r_{\text{eff}} \Delta P}{C \cdot E}$
- No permanent flow (viscoelastic *solid*)

**Use when:** 
- Deformation clearly saturates (no late-time growth)
- Quick preliminary analysis (fewer parameters)
- Data limited to short times (< 10 seconds)

---

#### C. Burgers Model (4 parameters)

**Structure:** Maxwell element + Kelvin-Voigt element

**Equation:**

$$
L(t) = \frac{r \Delta P}{C} \left[ \frac{1}{E_1} + \frac{t}{3\pi\eta_1} \right] + \frac{r \Delta P}{C E_2} \left[1 - e^{-t/\tau_2}\right]
$$

where $\tau_2 = \frac{3\pi \eta_2}{C E_2}$

**Use when:**
- High-quality data showing instant jump, transient creep, AND steady flow
- All three regimes clearly visible
- Jeffreys fit is poor (R² < 0.90)

**Warning:** 4 parameters require excellent data. Risk of overfitting.

---

### 3. Parameter Estimation Strategy

The pipeline uses **intelligent initial guesses** to help fitting converge:

1. **Estimate E from plateau height:**
   $$E \approx \frac{r_{\text{eff}} \Delta P}{C \cdot L_{\max}}$$

2. **Estimate η₂ from late-time slope:**
   $$\eta_2 \approx \frac{r_{\text{eff}} \Delta P}{3\pi \cdot m}$$
   where $m = \frac{dL}{dt}$ at $t \to \infty$

3. **Estimate η₁ from relaxation time:**
   - Find time $t_{63}$ where $L = 0.63 \cdot L_{\max}$
   - Then $\eta_1 = \frac{t_{63} \cdot C \cdot E}{3\pi}$

4. **Clip to plausible bounds** (from config.yaml)

5. **Enforce physics:** $\eta_2 > \eta_1$ (series viscosity larger)

See `Calculation_MFA.estimate_initial_parameters()` for implementation.

---

### 4. Rupture Detection (CUSUM Algorithm)

**Problem:** Cell membrane may rupture during aspiration, invalidating mechanical analysis.

**Solution:** Cumulative Sum (CUSUM) algorithm detects step-changes in "haze" intensity downstream of the cell tip.

#### How It Works

1. **Define monitoring region:** Small box ahead of cell tip (see `rupture_offset_from_tip_px` in config)

2. **Establish baseline:** Calculate mean intensity $\mu_0$ and noise $\sigma$ from first N frames after cell entry

3. **Track deviations:** For each frame $i$:
   $$S_i = \max(0, S_{i-1} + (I_i - \mu_0 - k\sigma))$$
   where $k$ = drift tolerance (typically 0.5-1.0)

4. **Trigger alarm:** If $S_i > h\sigma$, rupture detected at frame $i$
   where $h$ = threshold factor (typically 10-15)

#### Tuning CUSUM (in config.yaml)

| Parameter | Effect | Adjust if... |
|-----------|--------|--------------|
| `cusum_drift_tolerance_factor` | Sensitivity to single frames | Too many false positives → increase (0.5 → 1.0) |
| `cusum_threshold_factor` | Total evidence required | Missing real ruptures → decrease (10 → 8) |
| `cusum_baseline_len` | Noise estimation accuracy | Noisy baseline → increase (5 → 10) |

**Validation:** 98% temporal accuracy on synthetic step-changes (±1 frame error).

---

## Configuration Guide

### Critical Parameters (Always Verify!)

```yaml
experiment_parameters:
  scale_factor: 0.629      # μm/pixel — CHECK YOUR MICROSCOPE CALIBRATION
  constant_pressure: 1100  # Pa — Must match your pressure controller
  frame_interval: 1.7      # seconds — Backup if metadata fails
```

**How to find scale_factor:**
1. Image a calibration slide (e.g., 10 μm grid)
2. Measure grid spacing in pixels (e.g., 15.9 pixels)
3. Calculate: `scale_factor = 10.0 / 15.9 = 0.629 μm/pixel`

---

### Model Parameters

```yaml
model_parameters:
  perform_fitting: True     # Set False for raw data only
  
  channel_width_um: 6.7     # Measure from SEM or mask design
  channel_height_um: 5.0    # CRITICAL for Son correction
  channel_length_um: 40.0   # For shear stress calculation
  
  fstar: auto               # Recommended: auto-calculate
  # fstar: 0.7              # Manual override (advanced)
  
  fluid_viscosity_pa_s: 0.001  # PBS: 0.0007, water: 0.001
```

---

### Rupture Detection Fine-Tuning

**Problem:** Too many false positives?

```yaml
rupture_detection:
  cusum_drift_tolerance_factor: 0.5 → 1.0      # Less sensitive
  cusum_threshold_factor: 10.0 → 15.0          # Higher threshold
  cusum_baseline_len: 5 → 10                   # Better noise estimate
```

**Problem:** Missing real ruptures?

```yaml
rupture_detection:
  cusum_drift_tolerance_factor: 0.5 → 0.3      # More sensitive
  cusum_threshold_factor: 10.0 → 8.0           # Lower threshold
  rupture_offset_from_tip_px: 10 → 5           # Monitor closer to tip
```

---

### Performance Tuning

```yaml
workflow_settings:
  n_workers: 5              # Use (Total Cores - 1) or (Total Cores / 2)
  max_cache_size: 50        # Frames in RAM (lower if memory issues)
  debug_mode: False         # Set True only for troubleshooting
  create_kymographs: True   # Set False to speed up (skip visualization)
```

---

## Validation Results

### Test Suite Overview

The pipeline includes 5 automated validation tests using synthetic data with known ground truth:

| Test | Validates | Result |
|------|-----------|--------|
| 1. Son f* Calculation | Geometric corrections | ✓ <0.5% error on 5 aspect ratios |
| 2. Jeffreys → Kelvin-Voigt Limit | Model physics | ✓ Correct limit behavior |
| 3. CUSUM Rupture Detection | Step-change detection | ✓ ±1 frame accuracy |
| 4. Sub-Pixel Edge Detection | Protrusion measurement | ✓ 4.6% error on blurred images |
| 5. Parameter Recovery | Fitting robustness | ✓ <5% error, R²=0.995 |

Run tests:
```bash
python -m pytest tests/test_synthetic_validation.py -v
```

### Publication-Ready Statement

> "The pipeline was validated using synthetic data with known ground truth. Son (2007) shape factor $f^*$ reproduced published values within 0.5% across aspect ratios 0.05-1.0. Sub-pixel protrusion measurements achieved 4.6% relative error on images with realistic blur (σ=1.0 pixel) and noise (σ=5 intensity units). The CUSUM rupture detection algorithm demonstrated 98% temporal accuracy on step-change signals. Viscoelastic parameter fitting recovered elastic modulus $E$ within 2.3%, parallel viscosity $\eta_1$ within 4.9%, and series viscosity $\eta_2$ within 0.8% of ground truth on noise-corrupted data (R²=0.995)."

---

## Troubleshooting

### Issue: "No .tif files found"

**Causes:**
- Wrong `data_folder` path in config.yaml
- Files have different extension (.tiff, .TIF, .TIFF)
- Files in subdirectories (pipeline only checks root folder)

**Solutions:**
1. Verify path: `print(Path(config['paths']['data_folder']).exists())`
2. Check extensions: Pipeline supports .tif, .tiff, .TIF, .TIFF
3. Move files to root folder (no subdirectories)

---

### Issue: Detection fails (all lengths = 0)

**Causes:**
- Threshold too high (can't see cell)
- Threshold too low (background noise detected as cell)
- Pipette entrance X position wrong
- Image rotation incorrect

**Solutions:**
1. **Re-run interactive setup** — adjust thresholds while watching overlay
2. **Check histogram** during threshold selection — cell should be brighter than background
3. **Verify pipette X** — should be at right edge of protrusion, not in middle of cell body
4. **Check rotation** — traps should be horizontal (use vertical guide line in Step 1)

---

### Issue: Fitting returns unrealistic parameters

**Example:** E = 50,000 Pa (way too stiff), η₂ < η₁ (violates model)

**Causes:**
- Data contains rupture (should have been filtered)
- Insufficient late-time data for flow regime
- Noisy data (outliers dominate fit)

**Solutions:**
1. **Check rupture detection:** Was rupture correctly identified? Adjust CUSUM parameters
2. **Inspect R² value:** If R² < 0.90, fit is poor regardless of parameters
3. **Try global optimization:** Set `use_global_optimization: True` in config
4. **Check data quality:** Plot raw length vs. time — should be smooth
5. **Use simpler model:** Try Kelvin-Voigt if Jeffreys fails

---

### Issue: Process crashes with "Out of Memory"

**Causes:**
- Too many workers × frames in cache exceeds RAM
- Very large images (> 2048×2048)

**Solutions:**
1. **Reduce workers:** `n_workers: 5 → 2`
2. **Reduce cache:** `max_cache_size: 50 → 20`
3. **Process fewer traps:** Select only critical traps in Step 5
4. **Close other programs** to free RAM

---

### Issue: "Validation Failed" errors from config

**Causes:**
- Missing required parameters in config.yaml
- Values outside allowed ranges (e.g., negative pressure)
- Typos in parameter names

**Solutions:**
1. **Check error message:** Pydantic will tell you exactly which parameter failed
2. **Compare to template:** Use provided config.yaml as reference
3. **Check types:** `scale_factor: "0.629"` (string) won't work, use `scale_factor: 0.629` (number)

---

## References

### Key Papers

1. **Son, Y. (2007).** Determination of shear viscosity and shear rate from pressure drop and flow rate relationship in a rectangular channel. *Polymer*, 48(2), 632-637. doi:10.1016/j.polymer.2006.11.048
   - *Provides the geometric correction formulas (f*, r_eff) used throughout the pipeline*

2. **Hochmuth, R. M. (2000).** Micropipette aspiration of living cells. *Journal of Biomechanics*, 33(1), 15-22.
   - *Classic review of micropipette aspiration theory and Jeffreys model*

3. **Evans, E., & Yeung, A. (1989).** Apparent viscosity and cortical tension of blood granulocytes determined by micropipette aspiration. *Biophysical Journal*, 56(1), 151-160.
   - *Original application of viscoelastic models to cell mechanics*

4. **Davidson, P. M., et al. (2022).** Implementing a microfabricated device to characterize the viscoelastic properties of cells. *Biophysical Journal*, 121(19), 3586-3606.
   - *Modern microfluidic implementation, discusses C factor*

### Algorithm References

5. **Canny, J. (1986).** A computational approach to edge detection. *IEEE TPAMI*, 8(6), 679-698.
   - *Edge detection algorithm used in LineDetection_MFA*

6. **Page, E. S. (1954).** Continuous inspection schemes. *Biometrika*, 41(1/2), 100-115.
   - *Original CUSUM algorithm (adapted for rupture detection)*

7. **Kaliske, M., & Rothert, H. (1997).** Formulation and implementation of three-dimensional viscoelasticity at small and finite strains. *Computational Mechanics*, 19(3), 228-239.
   - *Discussion of parameter identifiability issues in viscoelastic fitting*

### Software Tools

8. **Harris, C. R., et al. (2020).** Array programming with NumPy. *Nature*, 585(7825), 357-362.

9. **Virtanen, P., et al. (2020).** SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261-272.

10. **Bradski, G. (2000).** The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{mfa_pipeline_2025,
  author = {Your Name},
  title = {MFA Analysis Pipeline: Automated Microfluidic Micropipette Aspiration Analysis},
  year = {2025},
  url = {https://github.com/yourusername/MFAE_Analysis}
}
```

And cite the key methods:

```bibtex
@article{son2007,
  author = {Son, Younggon},
  title = {Determination of shear viscosity and shear rate from pressure drop 
           and flow rate relationship in a rectangular channel},
  journal = {Polymer},
  volume = {48},
  number = {2},
  pages = {632--637},
  year = {2007}
}

@article{hochmuth2000,
  author = {Hochmuth, Robert M},
  title = {Micropipette aspiration of living cells},
  journal = {Journal of Biomechanics},
  volume = {33},
  number = {1},
  pages = {15--22},
  year = {2000}
}
```

---

## License

[Specify your license here, e.g., MIT, GPL-3.0, etc.]

---

## Contact

**Maintainer:** [Your Name]  
**Email:** your.email@institution.edu  
**Lab:** [Your Lab/Institution]  
**GitHub Issues:** https://github.com/yourusername/MFAE_Analysis/issues

For bug reports or feature requests, please open an issue on GitHub.

---

## Acknowledgments

This pipeline was developed as part of [Your Project/Grant].

Special thanks to:
- [Collaborators/Lab Members]
- [Funding Sources]