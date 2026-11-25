# MFAE Analysis

Image-analysis pipeline for **microfluidic aspiration (MFA)** and **electroporation (EP)**
experiments. For every trap in a time-lapse `.tif` sequence it tracks the length and shape of the
aspirated cell protrusion, detects membrane rupture, and (optionally) quantifies dye uptake, actin
distribution and viscoelastic parameters. A second stage pools many experiments into
condition-level statistics and figures.

```
.czi file / .tif folder ──► run_experiment.py ──► Output/<experiment_id>/      (one run per experiment)
                                                        │
              Output/Bulk_Analysis_results/ ◄── run_bulk_analysis.py   (pools every experiment in Output/)
```

You only ever run these two scripts. Everything else is the `mfae/` package they start:

```
run_experiment.py        analyse one experiment
run_bulk_analysis.py     pool many experiments
config/                  your experiment configs + defaults.yaml (pipeline settings)
mfae/                    the analysis code (see "For developers")
tools/                   small helpers (czi_info.py, plot_cell_status_overview.py)
docs/                    experiment naming rule
tests/                   automated tests (synthetic data)
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

All config files live in the `config/` folder. Two templates are the starting point for every
experiment:

| Template | For |
|---|---|
| `config/config_tiff_template.yaml` | A folder of exported `.tif` files |
| `config/config_czi_template.yaml` | A Zeiss `.czi` file |

Don't edit the templates themselves; make a copy per experiment (or per series of experiments
recorded with the same settings). `config/defaults.yaml` holds the pipeline settings and is the
same for everyone.

**Step 1 – create a config file** from the template, with your data folder filled in:

```bash
python run_experiment.py --init-config config/my_experiment.yaml --data-folder "D:/MFAE/260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-1ms-frame10"
```

This copies `config/config_tiff_template.yaml`, or `config/config_czi_template.yaml` when
`--data-folder` is a `.czi` file. The experiment name defaults to the folder's (or file's) name,
and `experiment_type` is set from its voltage. To start from one of your own configs instead,
add `--from config/<your config>.yaml`. You can also just copy a template by hand and fill in
`data_folder` and `experiment_id`.

A user config holds everything about one experiment, and nothing else:

| Section | Settings |
|---|---|
| Experiment type | `experiment_type: ep` (electroporation) or `asp` (aspiration only) |
| Experiment | data folder, experiment name, microscope scale, pressure, frame interval, max traps |
| Analyses (True / False) | dye uptake, actin, rupture detection, length smoothing, outlier rejection, viscoelastic fitting, kymographs; plus the channel names |
| Interactive setup | which frame is shown during setup, starting trap-box size |
| Computer | number of CPU cores |

Every one of these must be present (copy from the matching template if one is missing). Open the
file and check the values, in particular `scale_factor` (µm per pixel for your objective and
binning) and the channel settings.

**The pulse is read from the experiment name** (see [Naming experiment folders](#5-naming-experiment-folders)):
`…-100V-1ms-frame10` means a 100 V, 1 ms pulse at file number 10. Any voltage or duration works; you
never type them into the config. `experiment_type` must agree with the name (`asp` needs `0V`).

**Step 2 – check the setup** (opens no windows, writes no results):

```bash
python run_experiment.py config/my_experiment.yaml --check
```

This validates every setting and scans the images with the same code the analysis uses. It reports
how many frames each channel has, the frame interval, where the pulse falls, and whether the folder
name will be recognised by the bulk analysis. Fix anything marked `PROBLEM` before continuing.

**Step 3 – run the analysis:**

```bash
python run_experiment.py config/my_experiment.yaml
```

Results are written to `Output/<experiment_id>/`.

### Running from Spyder

The commands above are for a terminal. In Spyder, the pipeline needs to be told **which config
file to use**: pressing Run (F5) on `run_experiment.py` without one only prints the help text.
Add the config file to the file's run configuration once:

1. Open `run_experiment.py` in Spyder.
2. **Run → Configuration per file…** (Ctrl+F6).
3. Tick **Command line options** and enter the path to your config, e.g.
   `config/my_experiment.yaml`. Add options after it as needed, e.g.
   `config/my_experiment.yaml --check` or `config/my_experiment.yaml --fresh`.
4. Under **Working directory**, choose **The directory of the file being executed**, so that
   `config/...` is found.
5. Click **OK**, then press **F5**.

Spyder remembers this per file, so later runs only need F5. To analyse another experiment,
change the config path in the same dialog (and remove `--check` once the check passes).

Alternatively, type the same in the IPython console with `runfile`; `args` is everything that
would follow `python run_experiment.py` in a terminal:

```python
runfile('C:/GitHub/MFAE_Analysis/run_experiment.py', args='config/my_experiment.yaml --check', wdir='C:/GitHub/MFAE_Analysis')
```

Typing `python run_experiment.py ...` in the console gives `SyntaxError: invalid syntax`.

### Many experiments at once (batch)

Put the folders you want to analyse in a config, under `data_folder`, and run it with `--batch`.
Every experiment in those folders is found (at any depth): each `.czi` file for a CZI config, each
folder of `.tif` images for a TIFF config. Experiments that need other settings go in another
config with their own folders; run that one the same way.

The configs for your data are ready:

| Config | Folders | Settings |
|---|---|---|
| `config/config_symmetric_czi.yaml` | `MFAE\Symmetric_Efield` | symmetric microscope (.czi) |
| `config/config_asymmetric_tiff.yaml` | `MFAE\Asymmetric_Efield\5ms_100V` | asymmetric (.tif) |
| `config/config_asymmetric_100us_tiff.yaml` | `MFAE\Asymmetric_Efield\100us_100V` | asymmetric, 100 µs settings |
| `config/config_aspiration_tiff.yaml` | `MFAE\Aspiration` | aspiration only (.tif) |

In Spyder, one line per config:

```python
runfile('C:/GitHub/MFAE_Analysis/run_experiment.py', args='config/config_symmetric_czi.yaml --batch', wdir='C:/GitHub/MFAE_Analysis')
runfile('C:/GitHub/MFAE_Analysis/run_experiment.py', args='config/config_asymmetric_tiff.yaml --batch', wdir='C:/GitHub/MFAE_Analysis')
runfile('C:/GitHub/MFAE_Analysis/run_experiment.py', args='config/config_asymmetric_100us_tiff.yaml --batch', wdir='C:/GitHub/MFAE_Analysis')
runfile('C:/GitHub/MFAE_Analysis/run_experiment.py', args='config/config_aspiration_tiff.yaml --batch', wdir='C:/GitHub/MFAE_Analysis')
```

**One experiment** with the same config: give it with `--data-folder` instead of `--batch`. Its
results go to the same place the batch would put them (and the batch later skips it):

```python
runfile('C:/GitHub/MFAE_Analysis/run_experiment.py', args='config/config_asymmetric_tiff.yaml --data-folder "M:/tnw/bn/gk/NN/2_Data-Analysis/MFAE/Asymmetric_Efield/5ms_100V/260114_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-5ms-frame10"', wdir='C:/GitHub/MFAE_Analysis')
```

For a .czi experiment, `--data-folder` is the .czi file itself.

To add a folder, add a line under `data_folder` in the config:

```yaml
  data_folder:
    - 'M:\tnw\bn\gk\NN\2_Data-Analysis\MFAE\Asymmetric_Efield\5ms_100V'
    - 'M:\tnw\bn\gk\NN\2_Data-Analysis\MFAE\Asymmetric_Efield\10ms_100V'
```

(or give folders on the command line: `--batch "M:/.../5ms_100V" "M:/.../10ms_100V"`).

What happens:

1. **Setup of every experiment, one after another** (trap boxes, entrances, thresholds). Saved
   setups are reused, so experiments set up earlier don't ask again. Esc stops the batch; the
   setups done so far are kept.
2. **Analysis of all of them, without interaction**: leave it running (e.g. overnight).
3. A summary (done / skipped / failed per experiment) is printed and saved as
   `Output/<setup>/batch_summary_<date>_<time>.txt`. A failure in one experiment doesn't stop the rest.

The pulse and the experiment type (0 V = aspiration only) are read from each experiment's name;
names that don't follow the [naming rule](#5-naming-experiment-folders) are listed and skipped.

**Where results go.** `Output/<setup>/<experiment>/`, the setup being the folder's name
(`Symmetric_Efield`, `Aspiration`) or, for a condition folder such as `5ms_100V`, the setup folder
that holds it (`Asymmetric_Efield`). So the 5 ms and 100 µs configs both fill
`Output/Asymmetric_Efield/`, setups are never mixed, and the bulk analysis analyses each setup on
its own (with the aspiration experiments as controls, see [Bulk analysis](#6-bulk-analysis-many-experiments)).
`--setup NAME` sets the results folder name yourself.

**Adding data later:** run the same line again. Experiments that already have results are skipped;
only new ones are set up and analysed. `--redo` analyses everything again (reusing the saved
setups; add `--fresh` to redo the setups too). `--batch --check` checks every experiment without
opening windows.

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
| `--init-config FILE` | Create a new config from the TIFF or CZI template (or `--from OTHER.yaml`) |

Because `--data-folder` overrides the config, one config can be reused for a whole series of
experiments recorded with the same settings:

```bash
python run_experiment.py config/my_5ms_series.yaml --data-folder "D:/MFAE/260129_MDAMB231_WT_Chip1_Experiment4-1100Pa-100V-5ms-frame10"
```

### How settings work

Settings live in two files that **never share a setting**, so either can be rewritten without
affecting the other:

| File | Contains | Who edits it |
|---|---|---|
| Your user configs (copies of `config/config_tiff_template.yaml` or `config/config_czi_template.yaml`) | Exactly the user settings listed above. | You, per experiment |
| `config/defaults.yaml` | Pipeline settings: image processing, detection thresholds, fitting, figures. | Developers only |

The pipeline refuses to start if a user config is missing a user setting or contains a pipeline
setting (a typo also ends up here), or if `config/defaults.yaml` contains a user setting. The list of
user settings is `USER_KEYS` in `mfae/config/loader.py`.

- **To see exactly what a run will use:** `python run_experiment.py my.yaml --show-config`. Every
  value that doesn't come from `config/defaults.yaml` is marked with where it came from.
- **Every run saves its complete settings** to `Output/<experiment_id>/Run_Info/effective_config.yaml`. That
  file is a *full config* (it lists every setting and has no `experiment_type`), so you can rerun an
  old analysis with exactly the same settings even after the defaults have changed. Old full-length
  config files work the same way.

### Using .czi files directly

The pipeline can read a Zeiss `.czi` file instead of exported TIFFs. Start from the CZI template:

```bash
python run_experiment.py --init-config config/my_experiment.yaml --data-folder "D:/MFAE/260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-1ms-frame10.czi"
```

When `--data-folder` is a `.czi` file, `config/config_czi_template.yaml` is copied automatically, and the
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

Your choices are saved to `Output/<experiment_id>/Run_Info/session_cache.json`, so a re-run skips the setup
and goes straight to the analysis (see [The saved setup](#the-saved-setup-session_cachejson)).

### The saved setup (session_cache.json)

When the interactive setup is finished, everything you chose in it is written to
`Output/<experiment_id>/Run_Info/session_cache.json` (results from older versions kept it directly in
the experiment folder; it is still found there and moved). The next run of the same experiment (same results
folder) finds the file, skips the setup windows and goes straight to the analysis.

```json
{
  "experiment_id": "261119_MDAMB231_LifeAct_Chip1_Experiment2-1100Pa-100V-1ms-frame304",
  "threshold_scale": "membrane SNR, 255 = 40 sigma",
  "rotation_angle": 0.7,
  "trap_configs": [
    {"trap_index": 0, "roi_coords": [240, 291, 112, 246], "pip": 76.0,
     "thr_prot": 64.0, "thr_body": 70.0, "smooth": 3.0}
  ],
  "source_files": ["..."]
}
```

| Field | Meaning |
|---|---|
| `rotation_angle` | Image rotation [degrees] |
| `trap_index` | Trap number minus one (`0` = Trap 1) |
| `roi_coords` | Trap box `[top, bottom, left, right]` in pixels of the rotated image |
| `pip` | Pipette entrance, in pixels from the left edge of the trap box |
| `thr_prot`, `thr_body` | Protrusion and cell-body thresholds (0–255) |
| `smooth` | Faint-cell smoothing along the channel [px] (the `F` key) |
| `threshold_scale` | What the thresholds are measured in (see below) |
| `source_files` | The image files the setup was made on |

Only these interactive choices are stored. Every other setting is read from your config and
`config/defaults.yaml` on each run, so you can change e.g. `scale_factor` or rupture settings
and re-run without repeating the setup.

- **Redo the setup:** run with `--fresh` (or delete the file). The new setup overwrites it.
- **Edit by hand:** the file is plain text. You can change a threshold, the pipette position or
  the smoothing of one trap, or delete a trap's entry to leave it out. Keep the JSON valid
  (commas, brackets).
- **Thresholds and settings must match.** A threshold is only meaningful on the scale it was
  picked on. If `threshold_scale` doesn't match the current mask settings (`membrane_enhancement`,
  `membrane_snr_full_scale`), the file is ignored, a warning says why, and the setup runs again.
  Caches from before the contrast enhancement have no `threshold_scale` and count as the old
  CLAHE method, so they also trigger a new setup.
- **Changed data folder:** if the images differ from `source_files`, the log warns but the setup
  is still reused; check the results, or use `--fresh`.

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
4. If the protrusion mask runs on to the edge of the box through faint signal ahead of the tip
   (out-of-focus light or debris in the channel), it is cut where the signal along the channel
   stays below 40 % of the protrusion's own level (`protrusion_leak_fraction`). Otherwise single
   frames read the full box length. Masks that end inside the box are never cut. The tip is then
   placed on the protrusion's centre-line profile, where the intensity drops to 25 % of the way
   from the channel background ahead of the tip to the protrusion just inside it
   (`tip_level_fraction`; lower follows dimmer tips). If the profile is still above that level
   15 px beyond the mask edge (signal in the channel, not the tip), or the contrast is too low,
   the mask edge is used instead (`max_tip_refinement_px`).
5. A pixel must also be at least 30 % as bright (above background) as the brightest signal across
   the channel in its column (`membrane_relative_level`, 0 = off). Around a bright cell the
   out-of-focus halo is well above the noise but only 10-20 % of the cell's brightness, so without
   this the masks are wider and blobbier than the cell. Only 2 px either side along the channel
   count, so the dim shaft next to a bright spot in the protrusion is kept.
6. After thresholding, background enclosed by the cell is filled in (membrane stains are brightest
   at the rim, which would otherwise give ring-shaped masks), and small specks are removed.

**Dye and actin intensities are always measured on the raw pixels.** They only borrow the masks;
this also holds when actin is imaged in the same channel as the membrane.

The settings are in `config/defaults.yaml` under `image_processing` (`membrane_enhancement`,
`membrane_denoise_sigma`, `membrane_denoise_sigma_along`, `membrane_temporal_window`, `membrane_snr_full_scale`,
`membrane_fill_holes`, `membrane_max_hole_fraction`, `membrane_relative_level`). Set `membrane_enhancement: False` to return
to the older CLAHE-based segmentation; thresholds picked in one mode are not comparable with the
other.

## 4. Output

Each experiment gets one folder. It contains a `README.txt` that lists every file in it, says what
it shows, and explains the main columns, so you can hand the folder to someone else.

```
Output/<experiment_id>/
├── README.txt                         what every folder and file holds (start here)
├── 0_Overview/
│   ├── trap_map.png                   which trap is which
│   └── trap_summary.csv               one row per trap: analysed cell, first/last frame, why
│                                      tracking ended, protrusion at the pulse, what was analysed
├── 1_Protrusion_and_Shape/            membrane channel
│   ├── Figures/                       All_traps_protrusion_length.pdf, All_traps_membrane_kymographs.pdf,
│   │                                  All_traps_area_change.pdf, All_traps_body_solidity.pdf
│   ├── Data/                          Trap_XX_protrusion_and_shape.csv (every frame),
│   │                                  Trap_XX_membrane_kymograph.csv
│   └── Movies/                        Trap_XX_tracking.avi (masks, tip and length trace per frame)
├── 2_Dye_Uptake/                      dye channel                       (if enabled)
│   ├── Figures/                       All_traps_dye_uptake.pdf, _dye_kymographs.pdf, _dye_profiles_along_cell.pdf
│   ├── Data/                          Trap_XX_dye_uptake.csv, Trap_XX_dye_kymograph.csv
│   └── Movies/                        Trap_XX_dye_masks.avi
├── 3_Actin/                           actin channel                     (if enabled)
│   ├── Figures/                       All_traps_actin_kymographs.pdf, All_traps_actin_zones.pdf
│   ├── Data/                          Trap_XX_actin.csv, Trap_XX_actin_kymograph.csv
│   └── Movies/                        Trap_XX_actin_zones.avi, Trap_XX_actin_cortex.avi
├── 4_Mechanics/                       Flow_conditions.csv/.pdf (wall shear stress, flow rate),
│                                      Viscoelastic_fits.csv             (if fitting is on)
└── Run_Info/                          effective_config.yaml, session_cache.json, analysis.log
```

Frame numbers in `trap_summary.csv` and the movies count from 1 (as in Fiji); the `Frame` column
of the Data CSVs counts from 0. The masks in all three movies are the same LineDetection masks the
numbers were measured in. All folder and file names are defined in `mfae/files/output_layout.py`.

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

The bulk analysis reads this layout and the older ones (`Protrusion Length Detection/...`,
`Dye Uptake/Uptake traces/`, ..., and the flat one), with the old file names
(`trap_XX_detection_full.csv`, `Trap_XX_Uptake_Data.csv`, ...).

**Several cells in one trap.** A trap often holds more than one cell during a recording: a cell
passes through the channel and the next one arrives. Each trap is therefore split into cells, and
a cell ends when

- its protrusion tip stays at the left edge of the trap box for 3 frames (`left ROI`; the true
  length is unknown from then on, `rupture_detection.exit_at_roi_edge_frames`),
- the protrusion collapses for 3 frames (`slip`), or
- the length falls below half its recent level and stays there, i.e. the cell passed through and
  a new, shorter protrusion appears (`cell change`, `rupture_detection.cell_change_drop_ratio`).

After a cell ends, the next entry is searched from there. **The analysed cell is the one in the
trap at the pulse**, since that is the one that was electroporated; for aspiration-only runs it is
the longest-tracked cell. If the trap is empty at the pulse, the last cell before it is kept and the
log warns that it was not pulsed. All measurements (length, areas, rupture, dye uptake, actin)
cover only the analysed cell, from its own entry to its exit; other frames are empty (NaN). The log
lists every cell found, e.g. `Trap 4 held 2 cells (cell 1: frames 2-150 (cell change); cell 2:
frames 151-504). Analysing cell 2 (in the trap at the pulse).` The per-trap CSV has a `Cell`
column (which cell was in the trap in each frame), `Tracked` (the analysed cell's frames) and
`Exit_Reason`, and the tracking movie labels the other cells as "not analysed". If many cells leave
through the left edge, draw a wider trap box.

Rupture markers on the plots come from the automatic rupture detection, which is not reliable
enough to cut data on. All CSVs therefore contain every frame; treat the markers as hints.

## 5. Naming experiment folders

The pulse settings of a run and the condition groups of the bulk analysis are both read from the
experiment name, so every `experiment_id` must follow this pattern (details in
[`docs/experiment_naming.txt`](docs/experiment_naming.txt)):

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

After running `run_experiment.py` on each experiment:

```bash
python run_bulk_analysis.py
```

It reads the experiment results in `Output/` (use `--output-root PATH` if they live elsewhere),
groups the cells by condition (from the experiment names) and writes one folder per run, named by
date and time, with a `README.txt` that lists every file and what it shows.

**Each setup is analysed on its own, never pooled with another.** Every folder in `Output/` that
holds experiment results (e.g. `Symmetric_Efield/`, `Asymmetric_Efield/` from batch runs) is one
setup and gets its own `Bulk_Analysis_results/` inside it; experiments directly in `Output/` count
as one more setup.

**Aspiration-only controls are shared.** The experiments in `Output/Aspiration/` (no pulse, so no
field geometry) are added as the control group to every setup's analysis, so each setup's
pulse-vs-aspiration figures use them. Only aspiration-only (0 V) runs are taken from that folder.
`Aspiration/` also gets its own bulk analysis (aspiration-only mechanics). Use
`--controls FOLDER` for a different control folder, or `--controls none` for no shared controls.

```
Output/<setup>/Bulk_Analysis_results/<YYMMDD_HHMM>/
├── README.txt
├── 0_Tables/                  mechanics_results_all_traps.csv (one row per trap: fate, every fitted
│                              parameter and fit-quality flag), attrition_per_experiment.csv (cells
│                              found / analysed / excluded and why), Figure_Data/ (every plotted value)
├── 1_Aspiration_Mechanics/    aspiration-only cells: viscoelastic fits, treatments compared, controls
│   ├── <Treatment>/           one treatment (e.g. WT/, CytD/)
│   └── All_treatments/        treatments compared
├── 2_Before_Pulse/            pulsed vs aspiration-only cells, fitted before the pulse; fate vs mechanics
│   ├── Aspiration/  <pulse>/  one cohort (<pulse> = e.g. 100V_5ms)
│   └── All_conditions/        cohorts compared
├── 3_At_Pulse/                fate (intact / ruptured) per pulse and trap, uptake and actin around the pulse
│   ├── <pulse>/
│   └── All_conditions/
└── 4_Whole_Trace/             whole-recording fits, mean uptake traces, uptake per trap
    ├── <pulse>/
    └── All_conditions/
```

The treatment and pulse folders are made for the conditions in your data; folders a run writes
nothing to are removed. Figures can be redrawn from `0_Tables/Figure_Data/` without rerunning
(`mfae.bulk.figure_data.load_all(...)`). The layout is defined in `mfae/bulk/layout.py`.

In Spyder: `runfile('C:/GitHub/MFAE_Analysis/run_bulk_analysis.py', wdir='C:/GitHub/MFAE_Analysis')`.

**Marking cells by hand.** Add a tag to a trap's `Trap_XX_protrusion_and_shape.csv` file name
(e.g. `Trap_03_protrusion_and_shape_R.csv`) to tell the bulk analysis what happened to that cell:
`_R` ruptured after the pulse (still analysed); `_R0` ruptured before the pulse, `_DOA` / `_PI`
not viable, `_after` / `_POST` arrived after the pulse, `_MASK` / `_SHAPE` / `_IRREGULAR` detection
failed (these are excluded and only counted in the attrition table). Older
`trap_XX_detection_full_<TAG>.csv` names work the same way.

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
| The protrusion mask reaches the left edge of the box | The protrusion is longer than the trap box, so its true length can't be measured; tracking stops after 3 frames at the edge. Draw a wider ROI (`--fresh`). |
| `SyntaxError: invalid syntax` when running `python run_experiment.py ...` | The command was typed into a Python console. In Spyder use `runfile(...)` (see [Running from Spyder](#running-from-spyder)); otherwise run it in a terminal (Anaconda Prompt) from the repository folder. |
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

The code is one package, `mfae/`, with a sub-package per step. The two scripts in the project
folder only start it (`mfae.cli.main` and `mfae.bulk.run.main`). Optionally, `pip install -e .`
adds the same two commands, `mfae-experiment` and `mfae-bulk`, usable from any folder.

| Module | Role |
|---|---|
| `mfae/pipeline.py` | One experiment: setup, parallel per-trap workers, export (`MFAAnalysis`, `static_parallel_worker`) |
| `mfae/cli.py` | Command line of `run_experiment.py` (`--check`, `--init-config`, `--show-config`) |
| `mfae/batch.py` | `--batch`: find every experiment under a setup folder, setups first, then analyses |
| `mfae/paths.py` | Project, `config/` and default `Output/` folders |
| `mfae/config/schema.py` | Config validation (pydantic); every allowed setting is defined here |
| `mfae/config/loader.py` | Combining `config/defaults.yaml` with a user config |
| `mfae/config/naming.py` | The experiment-name rule (pulse settings, bulk condition groups) |
| `mfae/files/readers.py` | Finding images (.tif / .czi), channel matching, timestamps |
| `mfae/files/output_layout.py` | Folder layout and file names of the results (bulk readers use it too) |
| `mfae/imaging/trap_rois.py` | Interactive rotation, ROI and trap selection, cropping |
| `mfae/membrane/detection.py` | Masks, protrusion length, cells per trap, entry/exit and rupture detection, threshold windows |
| `mfae/membrane/kymograph.py`, `tracking_movie.py` | Membrane kymographs and per-trap tracking movies |
| `mfae/quantification/dye_uptake.py`, `actin.py` | Dye and actin channels, measured in the membrane masks |
| `mfae/mechanics/models.py`, `fitting.py` | Channel geometry and flow, viscoelastic models, fitting |
| `mfae/plotting/figures.py`, `multipanel.py` | Per-experiment figures (single-trap and all-trap) |
| `mfae/utils.py` | Shared helpers, including the membrane enhancement and mask generation |
| `mfae/bulk/` | Pooling experiments: `run.py` (pipeline), `loader.py`, `mechanics.py`, `layout.py`, `figure_data.py`, `plots/` |

Import modules by their package path (`from mfae.membrane import detection`); the scripts and
tests put the project folder on the path. To change where results are written, edit
`mfae/files/output_layout.py` only; the pipeline and the bulk analysis both read it.

When you add a config setting, define it in `mfae/config/schema.py` (unknown keys are rejected, so typos
can't be silently ignored). Then decide who owns it: a pipeline setting goes, with its value and an
explanation, in `config/defaults.yaml`; a user setting goes in `USER_KEYS` in `mfae/config/loader.py`
and in every user config, starting with the two templates (`config/config_tiff_template.yaml`,
`config/config_czi_template.yaml`).
