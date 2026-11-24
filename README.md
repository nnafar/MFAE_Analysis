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

**Step 1 – create a config file** for your experiment type:

```bash
python MFA_analysis.py --init-config configs/my_experiment.yaml --preset ep --data-folder "D:/MFAE/260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-5ms-frame10"
```

| Preset | Use for |
|---|---|
| `ep` (default) | Electroporation, 5 ms pulse, all analyses on |
| `5ms` | Electroporation, 5 ms pulse (rupture detection off) |
| `100us` | Electroporation, 100 µs pulse (rupture detection off) |
| `asp` | Aspiration-only controls, no pulse (rupture detection off) |

This writes a short file (about 40 lines) with only what changes between experiments: the data
folder, experiment name, microscope scale, pressure, frame interval, pulse frame and channel names.
`--data-folder` fills in the folder, and the experiment name defaults to the folder's name. Open the
file and check the values. See [How settings work](#how-settings-work) for everything else.

**Step 2 – check the setup** (opens no windows, writes no results):

```bash
python MFA_analysis.py configs/my_experiment.yaml --check
```

This validates every setting and scans the images with the same code the analysis uses. It reports
how many frames each channel has, the frame interval, where the pulse falls, and whether the folder
name will be recognised by the bulk analysis. Fix anything marked `PROBLEM` before continuing.

**Step 3 – run the analysis:**

```bash
python MFA_analysis.py configs/my_experiment.yaml
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
| `--init-config FILE --preset NAME` | Create a new short config |

Because `--data-folder` overrides the config, one config can be reused for a whole series of
experiments recorded with the same settings:

```bash
python MFA_analysis.py configs/ep_5ms.yaml --data-folder "D:/MFAE/260129_MDAMB231_WT_Chip1_Experiment4-1100Pa-100V-5ms-frame10"
```

### How settings work

A run combines three files; later ones win:

```
config/defaults.yaml          every setting, with the lab's tuned values and an explanation of each
  ⬑ config/presets/<preset>   the few settings that differ for one experiment type
     ⬑ your config file       paths, microscope, pulse ... plus anything you choose to change
```

- **To change any other setting**, copy it from `config/defaults.yaml` into your own file under the
  same section name, e.g.

  ```yaml
  image_processing:
    wall_clip_margin: 0.3
  ```

  Don't edit `config/defaults.yaml` for a single experiment; it changes the defaults for everyone.
- **To see exactly what a run will use:** `python MFA_analysis.py my.yaml --show-config`. Every
  value that doesn't come from the defaults is marked with the file that set it.
- **Every run saves its complete settings** to `Output/<experiment_id>/effective_config.yaml`. That
  file is itself a full config, so you can rerun an old analysis with exactly the same settings even
  after the defaults have changed.
- Old full-length config files (every setting listed, no `preset:`) still work unchanged.

## 3. The interactive setup

On the first run a series of windows guides you through the trap geometry. Each window shows its
own key hints; the common keys are:

| Key | Action |
|---|---|
| `Enter` | Confirm this step |
| `R` | Restart the setup from the beginning |
| `Esc` | Stop the analysis |
| `W` `A` `S` `D` | Adjust (rotation / ROI position / pipette position / threshold – depends on the step) |
| `J` / `L` | Decrease / increase display contrast (does not affect the analysis) |
| `I` / `K` | Decrease / increase display brightness (does not affect the analysis) |

The steps are:

1. **Rotate** the image so the channels are horizontal (`W`/`S` ±10°, `A`/`D` ±0.1°).
2. **Draw the ROI** around the first trap (drag with the mouse, fine-tune with `WASD`).
3. **Set the trap spacing** so the boxes line up with every trap.
4. **Mark the pipette entrance** (click).
5. **Select traps** to analyse (click to toggle).
6. **Per trap:** confirm the pipette line, then pick the detection threshold
   (`W`/`S` ±10, `A`/`D` ±1) so the mask follows the cell.

Your choices are saved to `Output/<experiment_id>/session_cache.json`, so a re-run skips the setup
and goes straight to the analysis. Use `--fresh` to set up again (or delete that file).

## 4. Output

```
Output/<experiment_id>/
├── <experiment_id>_analysis.log        full log of the run
├── session_cache.json                  saved interactive setup
├── trap_map.png                        which trap is which
├── Full protrusion detection/          per-trap CSV, every frame
├── Tracking visuals/
│   ├── <id>_Detection_Traces.pdf       length + haze, all traps
│   ├── <id>_Membrane_Kymographs.pdf    membrane kymograph, all traps
│   └── Trap_XX_Membrane_Kymograph.csv
├── Cell Morphology/                    aggregate shape metrics
├── Dye Uptake/                                                    (if enabled)
│   ├── <id>_Uptake_Kymographs.pdf      dye kymograph, all traps
│   ├── <id>_Diffusion_Profiles.pdf     dye profiles, all traps
│   └── Trap_XX_*.csv / *.avi           per-trap data and debug videos
├── Actin Quantification/                                          (if enabled)
│   ├── <id>_Actin_Kymographs.pdf       actin kymograph, all traps
│   ├── <id>_Actin_Zones.pdf            zone dynamics, all traps
│   └── Trap_XX_*.csv / *.avi           per-trap data and debug videos
└── Fitting results/                    shear analysis and viscoelastic fits   (if enabled)
```

Each plot type is one multipanel PDF with every trap as a panel. Panels share their axes and,
for kymographs, one colour scale, so traps can be compared at a glance. To also get one PDF per
trap (e.g. for a slide), set `plotting_parameters.save_individual_trap_plots: True`; the number
of panels per row is `plotting_parameters.multipanel_columns`.

Rupture markers on the plots come from the automatic rupture detection, which is not reliable
enough to cut data on. All CSVs therefore contain every frame; treat the markers as hints.

## 5. Naming experiment folders

The bulk analysis reads the experimental conditions from the folder name, so every
`experiment_id` must follow this pattern (details in
[`Bulk_Analysis/OtherInfo/Folder_naming.txt`](Bulk_Analysis/OtherInfo/Folder_naming.txt)):

```
YYMMDD_CellType_Treatment_Chip_Experiment-<P>Pa-<V>V-<D><ms|us|s>-frame<N>
260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-5ms-frame10      electroporated
260326_MDAMB231_WT_Chip2_Experiment283-1100Pa-0V-0ms-frame0       aspiration only
```

`--check` warns when a name does not match.

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
| Windows are too small on a high-resolution screen | Raise `workflow_settings.window_scale_factor` (e.g. 1.5). |
| The computer becomes sluggish | Lower `workflow_settings.n_workers`. |

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
| `config_loader.py`, `config/` | Combining defaults, presets and the user's file |
| `FileHandling_MFA.py` | Finding images, channel matching, timestamps |
| `ImageProcessing_MFA.py` | Interactive rotation, ROI and trap selection |
| `LineDetection_MFA.py` | Segmentation, protrusion length, entry/exit and rupture detection |
| `UptakeQuantification.py` / `ActinQuantification.py` | Dye and actin channels |
| `Calculation_MFA.py` / `Fitting_MFA.py` | Geometry, shear metrics and viscoelastic models |
| `Plotting_MFA.py`, `Plotting_Multipanel.py`, `Kymograph_MFA.py`, `Utils_MFA.py` | Figures and shared helpers |
| `Bulk_Analysis/` | Multi-experiment pooling, statistics and thesis figures |

When you add a config setting, add it to `config_schema.py` and document it in
`config/defaults.yaml`: unknown keys are rejected so typos
can't be silently ignored.
