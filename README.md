# MFAE Analysis

Image-analysis pipeline for **microfluidic aspiration (MFA)** and **electroporation (EP)**
experiments. For every trap in a time-lapse `.tif` sequence it tracks the length and shape of the
aspirated cell protrusion, detects membrane rupture, and (optionally) quantifies dye uptake, actin
distribution and viscoelastic parameters. A second stage, `Bulk_Analysis/`, pools many experiments
into condition-level statistics and figures.

```
raw .tif sequence ──► MFA_analysis.py ──► Output/<experiment_id>/   (one run per experiment)
                                              │
                         Bulk_Analysis/ ◄─────┘  (pools all experiments in Output/)
```

---

## 1. Installation

Requires **Python 3.10 or newer** (3.11–3.12 are tested).

```bash
git clone https://github.com/nnafar/MFAE_Analysis.git
cd MFAE_Analysis
python -m venv .venv
```

Activate the environment (Windows: `.venv\Scripts\activate`, macOS/Linux: `source .venv/bin/activate`), then:

```bash
pip install -r requirements.txt
```

The interactive windows use OpenCV and Tkinter. Tkinter ships with Python on Windows and macOS;
on Linux install it with your package manager (e.g. `sudo apt install python3-tk`).

## 2. Quick start (one experiment)

All config files live in the `config/` folder.

**Step 1 – create a config file** by copying `config/config.yaml`, the lab's standard experiment config,
with your data folder filled in:

```bash
python MFA_analysis.py --init-config config/my_experiment.yaml --data-folder "D:/MFAE/260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-1ms-frame10"
```

The experiment name defaults to the folder's name, and `experiment_type` is set from its voltage.
To start from one of your other configs instead, add `--from config/config_asp.yaml` (or any config file).
You can also just copy a config file by hand.

A user config holds everything about one experiment, and nothing else:

| Section | Settings |
|---|---|
| Experiment type | `experiment_type: ep` (electroporation) or `asp` (aspiration only) |
| Experiment | data folder, experiment name, microscope scale, pressure, frame interval, max traps |
| Analyses (True / False) | dye uptake, actin, rupture detection, length smoothing, outlier rejection, viscoelastic fitting, kymographs; plus the channel names |
| Interactive setup | which frame is shown during setup, starting trap-box size |
| Computer | number of CPU cores |

Every one of these must be present (copy from `config/config.yaml` if one is missing). Open the file and
check the values.

**The pulse is read from the experiment name** (see [Naming experiment folders](#5-naming-experiment-folders)):
`…-100V-1ms-frame10` means a 100 V, 1 ms pulse at file number 10. Any voltage or duration works; you
never type them into the config. `experiment_type` must agree with the name (`asp` needs `0V`).

**Step 2 – check the setup** (opens no windows, writes no results):

```bash
python MFA_analysis.py config/my_experiment.yaml --check
```

This validates every setting and scans the images with the same code the analysis uses. It reports
how many frames each channel has, the frame interval, where the pulse falls, and whether the folder
name will be recognised by the bulk analysis. Fix anything marked `PROBLEM` before continuing.

**Step 3 – run the analysis:**

```bash
python MFA_analysis.py config/my_experiment.yaml
```

Results are written to `Output/<experiment_id>/`.

### Command-line options

| Option | What it does |
|---|---|
| `--check` | Validate config and images, then exit |
| `--fresh` | Ignore the saved trap setup and redo the interactive steps |
| `--data-folder PATH` | Use this image folder instead of the one in the config |
| `--experiment-id NAME` | Name of the results folder (default: the data folder's name) |
| `--output-dir PATH` | Write results somewhere other than `Output/` |
| `--show-config` | Print every setting the run would use, and which file set it |
| `--debug` | Show detailed log messages |
| `--init-config FILE` | Create a new config by copying `config/config.yaml` (or `--from OTHER.yaml`) |

Because `--data-folder` overrides the config, one config can be reused for a whole series of
experiments recorded with the same settings:

```bash
python MFA_analysis.py config/config_5ms.yaml --data-folder "D:/MFAE/260129_MDAMB231_WT_Chip1_Experiment4-1100Pa-100V-5ms-frame10"
```

### How settings work

Settings live in two files that **never share a setting**, so either can be rewritten without
affecting the other:

| File | Contains | Who edits it |
|---|---|---|
| Your user configs (`config/config.yaml`, `config/config_5ms.yaml`, …) | Exactly the user settings listed above. | You, per experiment |
| `config/defaults.yaml` | Pipeline settings: image processing, detection thresholds, fitting, figures. | Developers only |

The pipeline refuses to start if a user config is missing a user setting or contains a pipeline
setting (a typo also ends up here), or if `config/defaults.yaml` contains a user setting. The list of
user settings is `USER_KEYS` in `config_loader.py`.

- **To see exactly what a run will use:** `python MFA_analysis.py my.yaml --show-config`. Every
  value that doesn't come from `config/defaults.yaml` is marked with where it came from.
- **Every run saves its complete settings** to `Output/<experiment_id>/effective_config.yaml`. That
  file is a *full config* (it lists every setting and has no `experiment_type`), so you can rerun an
  old analysis with exactly the same settings even after the defaults have changed. Old full-length
  config files work the same way.

### Using .czi files directly

The pipeline can read a Zeiss `.czi` file instead of exported TIFFs. Start from the CZI template:

```bash
python MFA_analysis.py --init-config config/my_experiment.yaml --data-folder "D:/MFAE/260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-1ms-frame10.czi"
```

When `--data-folder` is a `.czi` file, `config/config_czi.yaml` is copied automatically, and the
experiment name defaults to the file name without `.czi`, so name the file by the naming rule.
A CZI config differs from a TIFF one in two places:

```yaml
paths:
  image_format: czi          # tiff for a folder of exported .tif files
  data_folder: 'D:\MFAE'    # the .czi file, or its folder (file named after experiment_id)

dye_uptake_parameters:
  membrane_channel_number: 2 # channel numbers as Fiji shows them (dye first)
  dye_channel_number: 1
actin_parameters:
  channel_number: 2
```

Channels are chosen by number because the order in the CZI file (dye = 1, membrane = 2) is not the
order of the exported TIFFs (`C1` = membrane). TIFF configs use the name patterns instead; each
format's settings are refused in the other, so a TIFF config can't be applied to a CZI by mistake.

Frame numbers count from 1, as in Zen and Fiji, so `frame10` is the 10th frame. Frame times are
read from the file when it stores them; otherwise `frame_interval` is used. `--check` shows which,
the channel names, and the file's pixel size next to `scale_factor`. Only single-position,
single-plane time-lapses are supported. To see how a CZI file is organised without running the
pipeline, use `python tools/czi_info.py "D:/MFAE/<experiment>.czi"`.

## 3. The interactive setup

On the first run a series of windows guides you through the trap geometry. Each window shows its
own key hints; the common keys are:

| Key | Action |
|---|---|
| `Enter` | Confirm this step |
| `R` | Restart the setup from the beginning |
| `Esc` | Stop the analysis |
| `W` `A` `S` `D` | Adjust (rotation / ROI position / pipette position / threshold – depends on the step) |
| `F` | Threshold windows only: change the faint-cell smoothing (see below) |
| `J` / `L` | Decrease / increase display contrast (does not affect the analysis) |
| `I` / `K` | Decrease / increase display brightness (does not affect the analysis) |

The steps are:

1. **Rotate** the image so the channels are horizontal (`W`/`S` ±10°, `A`/`D` ±0.1°).
2. **Draw the ROI** around the first trap (drag with the mouse, fine-tune with `WASD`).
3. **Set the trap spacing** so the boxes line up with every trap.
4. **Mark the pipette entrance** (click).
5. **Select traps** to analyse (click to toggle).
6. **Per trap:** confirm the pipette line, then pick two thresholds: one for the **protrusion**
   (inside the channel, between the two guide lines) and one for the **cell body** (right of the
   pipette line). Drag the line in the histogram or use `W`/`S` ±10, `A`/`D` ±1 until the red mask
   follows the cell. The preview shows the mask the analysis will use (the analysis additionally takes the median over
   neighbouring frames, so its masks are a little steadier).

Your choices are saved to `Output/<experiment_id>/session_cache.json`, so a re-run skips the setup
and goes straight to the analysis. Use `--fresh` to set up again (or delete that file).

### Choosing the threshold

The threshold windows show a **contrast-enhanced membrane image**, not the raw image: background is
black and the cell is bright (see [How the masks are made](#how-the-masks-are-made)). The threshold
is shown both as a number (0–255) and as **noise units above background**, e.g.
`Threshold: 64 (~10.0 sigma above bg)`.

- The histogram has a background peak on the left and the cell on the right. Put the line in the
  gap between them; anywhere in that gap gives nearly the same mask. Around 5–15 sigma suits most
  cells.
- Too low (below ~3 sigma) and background noise joins the mask; too high and the dim protrusion tip
  is lost.
- The starting value is chosen automatically and is usually close.

**Faint cells (`F`).** For dim cells the image is smoothed more along the channel, which raises the
protrusion's signal-to-noise ratio without blurring its upper and lower edges. Each trap starts at
a level chosen from how bright its cell is (1 px for bright cells, up to 5 px for very faint ones);
`F` cycles 1 → 2 → 3 → 5 px and the preview updates. The current level is shown in the window
(`F: faint-cell smoothing (3 px)`). Use the lowest level at which the mask follows the protrusion:
more smoothing helps faint cells but slightly rounds the tip of bright ones. The level is saved per
trap together with the thresholds.

### How the masks are made

The membrane channel is used only to build two masks per frame: the **protrusion** and the
**cell body**. Because no intensity is measured from it, it is enhanced to make the cell outline
clear and consistent from frame to frame:

1. Each frame is replaced by the pixel-wise median of itself and its two neighbours, so noise at
   the threshold doesn't make the tip jitter or a faint protrusion flicker in and out
   (`membrane_temporal_window`, 1 = off). The image is then smoothed (1 px across the channel;
   along the channel by the per-trap level above).
2. In every frame, the background level and its noise are measured from the background peak of
   the histogram.
3. Each pixel becomes "noise units above background", so one threshold means the same thing in
   every frame, regardless of bleaching, exposure, or a very bright protrusion tip.
4. After thresholding, background enclosed by the cell is filled in (membrane stains are brightest
   at the rim, which would otherwise give ring-shaped masks), and small specks are removed.

**Dye and actin intensities are always measured on the raw pixels.** They only borrow the masks;
this also holds when actin is imaged in the same channel as the membrane.

The settings are in `config/defaults.yaml` under `image_processing` (`membrane_enhancement`,
`membrane_denoise_sigma`, `membrane_denoise_sigma_along`, `membrane_temporal_window`, `membrane_snr_full_scale`,
`membrane_fill_holes`, `membrane_max_hole_fraction`). Set `membrane_enhancement: False` to return
to the older CLAHE-based segmentation; thresholds picked in one mode are not comparable with the
other.

## 4. Output

```
Output/<experiment_id>/
├── <experiment_id>_analysis.log        full log of the run, including every trap's analysis
├── session_cache.json                  saved interactive setup
├── trap_map.png                        which trap is which
├── Protrusion Length Detection/
│   ├── <id>_Detection_Traces.pdf       length + haze, all traps
│   ├── <id>_Membrane_Kymographs.pdf    membrane kymograph, all traps
│   ├── Full protrusion detection/      trap_XX_detection_full.csv (every frame)
│   ├── Kymographs/                     Trap_XX_Membrane_Kymograph.csv
│   └── Movies/                         Trap_XX_Tracking.avi (masks, tip and length trace per frame)
├── Cell Morphology/                    aggregate shape metrics
├── Dye Uptake/                                                    (if enabled)
│   ├── <id>_*.pdf                      uptake kymographs, diffusion profiles, aggregates
│   ├── Uptake traces/                  Trap_XX_Uptake_Data.csv
│   ├── Uptake Kymographs/              Trap_XX_Uptake_Kymograph.csv
│   └── Uptake Movies/                  Trap_XX_Mask_Debug.avi
├── Actin Quantification/                                          (if enabled)
│   ├── <id>_*.pdf                      actin kymographs, zone dynamics
│   ├── Actin traces/                   Trap_XX_Actin_Data.csv
│   ├── Actin Kymographs/               Trap_XX_Actin_Kymograph.csv
│   └── Actin Movies/                   Trap_XX_Actin_Debug_Zones.avi / _Cortex.avi
└── Fitting results/                    shear analysis and viscoelastic fits   (if enabled)
```

Each plot type is one multipanel PDF with every trap as a panel. Panels share their axes and,
for kymographs, one colour scale, so traps can be compared at a glance. To also get one PDF per
trap (e.g. for a slide), set `plotting_parameters.save_individual_trap_plots: True` in
`config/defaults.yaml`; the number of panels per row is `plotting_parameters.multipanel_columns`
(both are pipeline settings).

The tracking movies (`workflow_settings.save_tracking_movies`, on by default) show the membrane
image with the protrusion (magenta) and body (cyan) masks, the pipette entrance, the tracked tip,
the rupture-monitoring box and the length trace with a cursor, so the tracking can be followed
frame by frame. They are MJPG `.avi` files; VLC plays them on any system. The status line shows
the frame, time, length and tracking state (pre-entry / tracking / exited / entry not confirmed),
plus PULSE and RUPTURED when they apply. Frame rate and size are set by
`workflow_settings.tracking_movie_fps` and `tracking_movie_width_px`.

Watching a trap's movie is the quickest way to check its measurement: if the length trace looks
wrong, the movie shows whether the mask lost the tip, picked up debris, or the cell left the trap.

The bulk analysis reads both this layout and the older flat one (CSVs directly in
`Full protrusion detection/`, `Dye Uptake/`, `Actin Quantification/`).

Rupture markers on the plots come from the automatic rupture detection, which is not reliable
enough to cut data on. All CSVs therefore contain every frame; treat the markers as hints.

## 5. Naming experiment folders

The pulse settings of a run and the condition groups of the bulk analysis are both read from the
experiment name, so every `experiment_id` must follow this pattern (details in
[`Bulk_Analysis/OtherInfo/Folder_naming.txt`](Bulk_Analysis/OtherInfo/Folder_naming.txt)):

```
YYMMDD_CellType_Treatment_Chip_Experiment-<P>Pa-<V>V-<D><ms|us|s>-frame<N>
260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-5ms-frame10      electroporated
260326_MDAMB231_WT_Chip2_Experiment283-1100Pa-0V-0ms-frame0       aspiration only
```

`--check` shows what it read from the name (e.g. `1100 Pa, 100 V, 5ms pulse at frame 10`) and warns
when a name does not match the rule or its pressure differs from `constant_pressure`. If a name
can't follow the rule, set `dye_uptake_parameters.has_pulse` and `pulse_frame` in your config
instead; the bulk analysis will skip that experiment.

## 6. Bulk analysis (many experiments)

After running `MFA_analysis.py` on each experiment:

```bash
python Bulk_Analysis/master_bulk_pipeline.py
```

```bash
python Bulk_Analysis/master_bulk_thesis.py
```

Both read every experiment folder in `Output/` and write to
`Output/Bulk_Analysis_results/<YYMMDD>/`. Use `--output-root PATH` if your per-experiment results
live elsewhere.

## 7. Troubleshooting

| Message | What to do |
|---|---|
| `... has N problem(s): - section.key: ...` | The named setting is invalid; the message says why. |
| `unknown setting (typo, or not supported ...)` | The key is misspelled or not a real setting. Compare with `config/defaults.yaml`. |
| `Config key '...' is ignored: ...` | The setting exists in older configs but has no effect; you can delete it. |
| `is not valid YAML` | Usually inconsistent indentation or an unclosed quote. Windows paths are safest in single quotes. |
| `No .tif files found` | Check `paths.data_folder`. |
| `Dye analysis is enabled but no file names contain 'C2'` | Set `dye_channel_pattern` to the text that identifies your dye images. |
| The setup windows don't appear on a re-run | A saved setup is being reused; add `--fresh`. |
| Windows are too small on a high-resolution screen | Raise `workflow_settings.window_scale_factor` (e.g. 1.5) in `config/defaults.yaml`. |
| The computer becomes sluggish | Lower `workflow_settings.n_workers` in your config. |
| A faint cell is barely captured in the threshold window | Press `F` for more smoothing, then lower the threshold (stay above ~3 sigma). |
| The mask includes background noise | Raise the threshold, or press `F` to reduce smoothing if it is high. |
| The protrusion mask reaches the left edge of the box | The protrusion is longer than the trap box, so its true length can't be measured (lengths close to the channel length are treated as clamped and don't count for entry). Draw a wider ROI (`--fresh`). |
| `Entry not confirmed for this trap` in the log | No valid protrusion was found for 3 frames in a row, so rupture detection is off for that trap. Watch its tracking movie. If the protrusion is tracked fine but longer than 0.9 × `channel_length_um` (36 µm by default), check `scale_factor`: `--check` shows the file's pixel size next to it. |
| A tracking movie won't play | Open it in VLC, which plays MJPG `.avi` files on any system. |

## 8. For developers

```bash
pip install -r requirements-dev.txt
```

```bash
pytest
```

The tests use synthetic data and need no microscope images; they also run on GitHub Actions for
every push (`.github/workflows/tests.yml`).

| File | Role |
|---|---|
| `MFA_analysis.py` | Entry point and orchestration (command line, setup, parallel workers, export) |
| `config_schema.py` | Config validation (pydantic); every allowed setting is defined here |
| `config_loader.py` | Combining `config/defaults.yaml` with a user config |
| `config/` | `defaults.yaml` (pipeline settings) and the user configs |
| `FileHandling_MFA.py` | Finding images, channel matching, timestamps |
| `ImageProcessing_MFA.py` | Interactive rotation, ROI and trap selection |
| `LineDetection_MFA.py` | Segmentation, protrusion length, entry/exit and rupture detection, threshold windows |
| `TrackingMovie_MFA.py` | Per-trap tracking movies |
| `output_layout.py` | Folder layout of the results (single source of truth; bulk readers use it too) |
| `UptakeQuantification.py` / `ActinQuantification.py` | Dye and actin channels |
| `Calculation_MFA.py` / `Fitting_MFA.py` | Geometry, shear metrics and viscoelastic models |
| `Plotting_MFA.py`, `Plotting_Multipanel.py`, `Kymograph_MFA.py` | Figures and kymographs |
| `Utils_MFA.py` | Shared helpers, including the membrane enhancement and mask generation |
| `Bulk_Analysis/` | Multi-experiment pooling, statistics and thesis figures |

To change where results are written, edit `output_layout.py` only; the pipeline and the bulk
analysis both read it.

When you add a config setting, define it in `config_schema.py` (unknown keys are rejected, so typos
can't be silently ignored). Then decide who owns it: a pipeline setting goes, with its value and an
explanation, in `config/defaults.yaml`; a user setting goes in `USER_KEYS` in `config_loader.py`
and in every user config, starting with `config/config.yaml`.
