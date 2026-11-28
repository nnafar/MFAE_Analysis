# MFAE Analysis

Image-analysis pipeline for **microfluidic aspiration (MFA)** and **electroporation (EP)**
experiments. For every trap in a time-lapse recording it tracks the length and shape of the
aspirated cell protrusion, detects membrane rupture, and (optionally) quantifies dye uptake, actin
distribution and viscoelastic parameters. A second stage pools many experiments into
condition-level statistics and figures.

```
.czi file / .tif folder ──► run_experiment.py ──► Output/<setup>/<experiment>/     (one per experiment)
                                                        │
       Output/<setup>/Bulk_Analysis_results/ ◄── run_bulk_analysis.py   (pools the experiments)
```

You only ever run these two scripts. Everything else is the `mfae/` package they start.

**New here?** Read sections 1–4 in order: install, run your first analysis, go through the setup
windows, check the results. The rest you can look up when you need it. Every command is given for a
terminal and for Spyder (see [How to run a command](#how-to-run-a-command)).

## Contents

1. [Installation](#1-installation)
2. [Your first analysis](#2-your-first-analysis)
3. [The setup windows](#3-the-setup-windows)
4. [Checking the results](#4-checking-the-results)
5. [Redoing an analysis](#5-redoing-an-analysis)
6. [Pooling experiments (bulk analysis)](#6-pooling-experiments-bulk-analysis)
7. [Naming experiments](#7-naming-experiments)
8. [Troubleshooting](#8-troubleshooting)
9. [Reference](#9-reference): all commands, options, configs and settings, `.czi` files, how the
   masks and the length are measured
10. [For developers](#10-for-developers)

```
run_experiment.py        analyse experiments
run_bulk_analysis.py     pool many experiments
config/                  your configs: config_asymmetric.yaml, config_symmetric.yaml
  pipeline/              files you don't edit: defaults.yaml (pipeline settings),
                         template_tiff.yaml / template_czi.yaml (copied for a new setup)
mfae/                    the analysis code
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

The setup windows use OpenCV and Tkinter. Tkinter ships with Python on Windows and macOS; on Linux
install it with your package manager (e.g. `sudo apt install python3-tk`).

---

## 2. Your first analysis

### How to run a command

Every command in this README is given in two forms. Use whichever fits how you work:

Terminal (Windows, macOS, Linux; built-in terminal of VS Code, PyCharm, ...):

```bash
python run_experiment.py config/config_asymmetric.yaml --batch --check
```

Spyder (IPython console):

```
%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --check" --wdir
```

Both do the same: run the script `run_experiment.py` with the config file and options after it.

**Terminal** (Command Prompt, PowerShell, Anaconda Prompt, the Terminal app on macOS/Linux, or the
built-in terminal of VS Code, PyCharm and other editors). Go to the repository folder and activate
the environment once per terminal window:

```bash
cd C:/GitHub/MFAE_Analysis
```

```bash
.venv\Scripts\activate
```

(macOS/Linux: `source .venv/bin/activate`; with Anaconda, `conda activate <your environment>`.)
Then paste the terminal line.

**Spyder.** Paste the Spyder line into the IPython console. Everything inside `--args "..."` is
passed to the script; options always go **inside** the quotes, since anything after `--wdir` is
read by Spyder, not by the script. `--wdir` runs it from the repository folder, so `config/...` is
found. Change `C:/GitHub/MFAE_Analysis` if your copy is elsewhere.

A terminal line typed into a Python console (Spyder's console, a Jupyter notebook, or after typing
`python`) gives `SyntaxError: invalid syntax`; use the Spyder line there.

<details>
<summary>Other ways to run from Spyder (F5, <code>runfile(...)</code>)</summary>

Pressing Run (F5) on `run_experiment.py` without a config only prints the help text. To use F5,
add the arguments to the file's run configuration once:

1. Open `run_experiment.py` in Spyder.
2. **Run → Configuration per file…** (Ctrl+F6).
3. Tick **Command line options** and enter everything after `python run_experiment.py`, e.g.
   `config/config_asymmetric.yaml --batch`.
4. Under **Working directory**, choose **The directory of the file being executed**.
5. Click **OK**, then press **F5**. Spyder remembers this per file.

The function form works too; `args` is everything after `python run_experiment.py`:

```python
runfile('C:/GitHub/MFAE_Analysis/run_experiment.py', args='config/config_asymmetric.yaml --batch', wdir='C:/GitHub/MFAE_Analysis')
```
</details>

### Step 1: pick the config for your microscope

There is **one config per microscope**, in the `config/` folder:

| Config | For | Data folders in it | Results |
|---|---|---|---|
| `config/config_asymmetric.yaml` | Asymmetric microscope: exported `.tif` folders (asymmetric E-field and aspiration controls) | `MFAE\Asymmetric_Efield` (all conditions), `MFAE\Aspiration` | `Output/Asymmetric_Efield/`, `Output/Aspiration/` |
| `config/config_symmetric.yaml` | Symmetric microscope: Zeiss `.czi` files (symmetric E-field) | `MFAE\Symmetric_Efield` | `Output/Symmetric_Efield/` |

The same config serves every condition: pulsed or aspiration-only, and the pulse voltage, duration
and frame, are read from each experiment's name (see [Naming experiments](#7-naming-experiments)).
The examples below use the asymmetric config; swap the name for symmetric data. You only need a new
config for data recorded with different settings (another objective, pressure or channel order),
see [Configs and settings](#93-configs-and-settings).

### Step 2: organise your data folders and list them

**Put each setup in its own folder.** A *setup* is one kind of experiment whose results are
compared with each other: one microscope and field geometry. Each setup folder gets its own
results folder and its own bulk analysis, and setups are never pooled. The lab's layout:

```
MFAE/
├── Asymmetric_Efield/          one setup (asymmetric microscope)
│   ├── 5ms_100V/               condition folders: optional, any number
│   │   ├── 260114_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-5ms-frame10/
│   │   └── ...
│   └── 100us_100V/
├── Symmetric_Efield/           another setup (symmetric microscope, .czi files)
│   └── 261119_MDAMB231_LifeAct_Chip1_Experiment2-1100Pa-100V-1ms-frame304.czi
└── Aspiration/                 aspiration-only (0 V) controls, shared by every setup (optional)
    └── 260326_MDAMB231_WT_Chip2_Experiment283-1100Pa-0V-0ms-frame0/
```

The rules:

- **The setup folder's name is only a label**: it becomes the results folder
  (`Output/Asymmetric_Efield/`). Call it what you like. The pipeline doesn't read "symmetric" or
  "asymmetric" from it, and nothing in the analysis depends on it.
- **`Aspiration` is the one name with a meaning.** The bulk analysis adds the 0 V experiments in
  `Output/Aspiration/` as controls to every setup. If your controls are in a folder with another
  name, tell the bulk analysis with `--controls <name>` (see
  [Pooling experiments](#6-pooling-experiments-bulk-analysis)).
- **What is pulsed and what is a control comes from each experiment's name** (`0V` = aspiration
  only), not from the folder (see [Naming experiments](#7-naming-experiments)).
  Aspiration-only experiments can also sit inside a setup folder; they are that setup's controls.
- **Condition folders** inside a setup folder are optional and can have any name, as long as you
  list the **setup** folder. If you list a condition folder on its own instead, name it by its
  pulse (`5ms_100V`, or `100V_5ms`, `5 ms 100 V`) so its results still go to its setup. A folder
  that mentions a pulse but has other text too (e.g. `5ms_100V_new`) becomes a setup of its own;
  the run warns about that.

**Only one setup?** That works the same. Put your experiments in one folder (with or without
condition folders), list it, and the bulk analysis analyses that one setup. Without an
`Aspiration` folder, it says so in the log and uses the 0 V experiments inside the setup as
controls, if there are any.

**List the folders** under `data_folder` in the config, one line each:

```yaml
  data_folder:
    - 'M:\tnw\bn\gk\NN\2_Data-Analysis\MFAE\Asymmetric_Efield'
    - 'M:\tnw\bn\gk\NN\2_Data-Analysis\MFAE\Aspiration'
```

Every experiment in those folders is found, at any depth: each `.czi` file for a CZI config, each
folder of `.tif` images for a TIFF config. You can also give folders on the command line instead,
after `--batch`: `--batch "E:/Data/MFAE/Aspiration"`. A symmetric `.czi` file placed in an
asymmetric setup folder would be pooled with the asymmetric data. Nothing checks this, so keep each
setup's data in its own folder.

### Step 3: check everything first

Terminal (Windows, macOS, Linux; built-in terminal of VS Code, PyCharm, ...):

```bash
python run_experiment.py config/config_asymmetric.yaml --batch --check
```

Spyder (IPython console):

```
%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --check" --wdir
```

This opens no windows and writes no results. It validates every setting and scans the images of
every experiment with the same code the analysis uses, and reports for each: the number of frames
per channel, the frame interval, what it read from the name (e.g. `1100 Pa, 100 V, 5ms pulse at
frame 10`) and where the pulse falls. Fix anything marked `PROBLEM` before continuing.

### Step 4: run the analysis

Terminal (Windows, macOS, Linux; built-in terminal of VS Code, PyCharm, ...):

```bash
python run_experiment.py config/config_asymmetric.yaml --batch
```

Spyder (IPython console):

```
%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch" --wdir
```

What happens:

1. **Setup of every new experiment, one after another**: you draw the trap boxes, mark the
   entrances and pick thresholds (see [The setup windows](#3-the-setup-windows)). Esc stops the
   batch; the setups done so far are kept.
2. **Analysis of all of them, without further questions.** You can leave it running (e.g. overnight).
3. **A summary** (done / skipped / failed per experiment) is printed and saved as
   `Output/<setup>/batch_summary_<date>_<time>.txt`. A failure in one experiment doesn't stop the rest.

Results go to `Output/<setup>/<experiment>/`. The setup is the listed folder's name
(`Symmetric_Efield`, `Aspiration`) or, for a condition folder such as `5ms_100V`, the setup folder
that holds it (`Asymmetric_Efield`). So setups are never mixed, and the bulk analysis treats each
on its own. `--setup NAME` sets the results folder name yourself.

Experiment names that don't follow the [naming rule](#7-naming-experiments) are listed and skipped.

### Adding data later

Run the same line again. Experiments that already have results are skipped; only new ones are set
up and analysed.

### Only some experiments

| To | Add to the command (in Spyder: inside the quotes) |
|---|---|
| Only experiments whose name contains a text | `--only Experiment4` |
| A folder that isn't listed in the config | `--batch "E:/Data/MFAE/Aspiration"` |
| One experiment by its folder (or `.czi` file), instead of `--batch` | `--data-folder "M:/.../260114_MDAMB231_WT_Chip1_Experiment4-1100Pa-100V-5ms-frame10"` |

With `--data-folder`, results go to the same place the batch would put them, and a later batch skips
that experiment. For a `.czi` experiment, `--data-folder` is the `.czi` file itself.

---

## 3. The setup windows

For each new experiment a series of windows guides you through the trap geometry. Each window shows
its own key hints.

**The steps:**

1. **Rotate** the image so the channels are horizontal (`W`/`S` ±10°, `A`/`D` ±0.1°).
2. **Draw the ROI** around the first trap (drag with the mouse, fine-tune with `WASD`).
3. **Set the trap spacing** so the boxes line up with every trap.
4. **Mark the pipette entrance** (click).
5. **Select traps** to analyse (click to toggle).
6. **Per trap:** confirm the pipette line, then pick two thresholds: one for the **protrusion**
   (inside the channel, between the two guide lines) and one for the **cell body** (right of the
   pipette line). Drag the line in the histogram or use `W`/`S` ±10, `A`/`D` ±1 until the red mask
   follows the cell. The preview shows the mask the analysis will use (the analysis additionally
   takes the median over neighbouring frames, so its masks are a little steadier).

**Keys:**

| Key | Action |
|---|---|
| `Enter` | Confirm this step |
| `B` (or `Backspace`) | Entrance and threshold windows: go back one window (body threshold → protrusion threshold → entrance → the previous trap). Each window opens at the value you chose before, so `Enter` keeps it |
| `R` | Entrance and threshold windows: start again at the first trap (trap boxes kept, earlier values pre-filled). Rotation / trap-box windows: restart the setup from the beginning |
| `Esc` | Stop the analysis |
| `W` `A` `S` `D` | Adjust (rotation / ROI position / pipette position / threshold, depending on the step) |
| `F` | Threshold windows only: change the faint-cell smoothing (see below) |
| `J` / `L` | Decrease / increase display contrast (does not affect the analysis) |
| `I` / `K` | Decrease / increase display brightness (does not affect the analysis) |

Your choices are saved, so a re-run skips the setup and goes straight to the analysis (see
[The saved setup](#the-saved-setup-session_cachejson)).

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

---

## 4. Checking the results

Each experiment gets one folder. It contains a `README.txt` that lists every file in it, says what
it shows and explains the main columns, so you can hand the folder to someone else.

```
Output/<setup>/<experiment>/
├── README.txt                         what every folder and file holds (start here)
├── 0_Overview/
│   ├── trap_map.png                   which trap is which
│   └── trap_summary.csv               one row per trap: analysed cell, first/last frame, why
│                                      tracking ended, protrusion at the pulse, what was analysed
├── 1_Protrusion_and_Shape/            membrane channel
│   ├── Figures/                       All_traps_protrusion_length.pdf, All_traps_membrane_kymographs.pdf,
│   │                                  All_traps_area_change.pdf, All_traps_body_solidity.pdf
│   ├── Data/
│   │   ├── Traces/                    Trap_XX_protrusion_trace.csv (length, areas, solidity; every frame)
│   │   └── Kymographs/                Trap_XX_membrane_kymograph.csv
│   └── Movies/                        Trap_XX_tracking.avi (masks, tip and length trace per frame)
├── 2_Dye_Uptake/                      dye channel                       (if enabled)
│   ├── Figures/                       All_traps_dye_uptake.pdf, _dye_kymographs.pdf, _dye_profiles_along_cell.pdf
│   ├── Data/Traces/                   Trap_XX_dye_uptake.csv
│   ├── Data/Kymographs/               Trap_XX_dye_kymograph.csv
│   └── Movies/                        Trap_XX_dye_masks.avi
├── 3_Actin/                           actin channel                     (if enabled)
│   ├── Figures/                       All_traps_actin_kymographs.pdf, All_traps_actin_zones.pdf
│   ├── Data/Traces/                   Trap_XX_actin.csv
│   ├── Data/Kymographs/               Trap_XX_actin_kymograph.csv
│   └── Movies/                        Trap_XX_actin_zones.avi, Trap_XX_actin_cortex.avi
├── 4_Mechanics/                       Flow_conditions.csv/.pdf (wall shear stress, flow rate),
│                                      Viscoelastic_fits.csv             (if fitting is on)
└── Run_Info/                          effective_config.yaml, session_cache.json, analysis.log,
                                       cell_annotations.csv (your tags, see below)
```

**Start with the figures and the tracking movies.** Each plot type is one multipanel PDF with every
trap as a panel; panels share their axes and, for kymographs, one colour scale, so traps can be
compared at a glance. If a trap's length trace looks wrong, its tracking movie shows why: whether
the mask lost the tip, picked up debris, or the cell left the trap.

The tracking movies show the membrane image with the protrusion (magenta) and body (cyan) masks,
the pipette entrance, the tracked tip (yellow), the rupture-monitoring box (red) and the length
trace with a cursor. The status line shows the frame, time, length and tracking state (pre-entry /
tracking / exited / entry not confirmed), plus PULSE and RUPTURED when they apply. They are MJPG
`.avi` files; VLC plays them on any system. The masks in all three movies (tracking, dye, actin)
are the same masks the numbers were measured in.

**Frame numbers** in `trap_summary.csv` and the movies count from 1 (as in Fiji); the `Frame`
column of the Data CSVs counts from 0.

**Several cells in one trap.** A trap often holds more than one cell during a recording: a cell
passes through the channel and the next one arrives. Each trap is therefore split into cells. A
cell ends when

- its protrusion tip stays at the left edge of the trap box for 3 frames (`left ROI`; the true
  length is unknown from then on),
- the protrusion collapses for 3 frames (`slip`), or
- the length falls below half its recent level and stays there, i.e. the cell passed through and
  a new, shorter protrusion appears (`cell change`).

**The analysed cell is the one in the trap at the pulse**, since that is the one that was
electroporated; for aspiration-only runs it is the longest-tracked cell. If the trap is empty at
the pulse, the last cell before it is kept and the log warns that it was not pulsed. All
measurements (length, areas, rupture, dye uptake, actin) cover only the analysed cell, from its own
entry to its exit; other frames are empty (NaN). The log lists every cell found, e.g.
`Trap 4 held 2 cells (cell 1: frames 2-150 (cell change); cell 2: frames 151-504). Analysing cell 2
(in the trap at the pulse).` The trace CSV has a `Cell` column (which cell was in the trap in each
frame), `Tracked` (the analysed cell's frames) and `Exit_Reason`, and the tracking movie labels
the other cells as "not analysed". If many cells leave through the left edge, draw a wider trap box.

**Rupture markers** on the plots come from the automatic rupture detection, which is not reliable
enough to cut data on. All CSVs therefore contain every frame; treat the markers as hints.

### Tagging the cells (review)

The pipeline can't reliably tell a ruptured, dead or badly segmented cell from a good one, so you
decide, while watching the tracking movies. Your tags go into `Run_Info/cell_annotations.csv` of
the experiment, one row per trap. The bulk analysis reads them from there, and **a re-analysis
keeps them**.

**Review the movies and tag with one key per trap.** No config and no raw data are needed, only the
results:

Terminal (Windows, macOS, Linux; built-in terminal of VS Code, PyCharm, ...):

```bash
python run_experiment.py --review --only LifeAct_Chip1_Experiment5
```

Spyder (IPython console):

```
%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "--review --only LifeAct_Chip1_Experiment5" --wdir
```

A window plays each trap's tracking movie (from just before the analysed cell arrives, on a loop),
starting at the first trap without a tag. Above the movie it shows the trap, the analysed cell and
its frames, the current tag with its meaning, and the pipeline's hint. Every change is saved
immediately, so you can stop at any time and continue later.

**Per trap: press the tag key(s), then `Enter`.** For a good cell, just press `Enter`.

| Key | Tag | Meaning | In the bulk analysis |
|---|---|---|---|
| `Enter` on an untagged trap | OK | Reviewed, nothing wrong | Analysed |
| `R` | R | **R**uptured (after the pulse) | Analysed (as ruptured) |
| `B` | R0 | Ruptured **b**efore the pulse | Excluded |
| `D` | DOA | **D**ead on arrival (not viable) | Excluded |
| `R` + `D` | R_DOA | Ruptured after the pulse and dead | Excluded |
| `B` + `D` | R0_DOA | Ruptured before the pulse and dead | Excluded |
| `P` | PI | **P**I positive (not viable) | Excluded |
| `L` | after | **L**ate: arrived after the pulse | Excluded |
| `M` / `S` / `I` | MASK / SHAPE / IRREGULAR | Detection failed: **m**ask / **s**hape / **i**rregular | Excluded |
| `C` or `Delete` | (none) | **C**lear the tag | Analysed (untagged) |

`R`, `B` and `D` combine, in either order; pressing a key again removes that part (so a wrong key
is undone by pressing it again). `P`, `L`, `M`, `S`, `I` stand alone and replace what was set.

| Key | Does |
|---|---|
| `Enter` | Accept the tag (an untagged trap becomes OK) and go to the next trap |
| `N` or `→` | Next trap without tagging (skip) |
| `Backspace` or `←` | Previous trap |
| `Space` | Pause / play |
| `,` `.` | One frame back / forward |
| `[` `]` | Slower / faster |
| `Q` or `Esc` | Quit (everything is already saved) |

The arrow keys depend on how your OpenCV draws its windows; `N` and `Backspace` always work. If an
arrow key does nothing, the window shows the code it sent; pass that number on so it can be added.

Add `--traps 3,6` to review only some traps. Without `--only`, every analysed experiment in
`Output/` is reviewed.

**Or edit the table in a spreadsheet.** `cell_annotations.csv` opens in Excel:

| Trap | Analysed_cell | First_frame | Last_frame | Tag | Note | Pipeline_hint | Check |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 67 | 600 | OK | | | |
| 3 | 2 | 265 | 314 | R | PI after pulse | rupture? frame 302 (Intensity Step); 3 cells in trap | |
| 6 | 2 | 53 | 600 | MASK | wall lines | | |

- **You fill in** `Tag` (the tags above; case doesn't matter) and
  `Note`. An empty `Tag` means not reviewed, and counts as intact. You can add columns of your own;
  they are kept.
- **The pipeline fills in** which cell was analysed, so you know which cell you are judging (the
  same numbers as in the movie), plus `Pipeline_hint`: what it suspects (automatic rupture, entry
  not confirmed, no cell at the pulse, several cells in the trap). The hint is never used as a tag.
- **When the experiment is analysed again**, your tags and notes are kept, and the cell and hint
  columns are updated. If a different cell is analysed in a trap than the one you tagged,
  `Check` says `CHECK: analysed cell changed (tag was set for frames ...)` and the log warns. Look at
  the trap again (`--review --traps N`), which clears the message, or delete the text yourself.
- **Close the file in Excel** before analysing or reviewing: Windows locks open files. If it is
  open, the log says so and your tags stay as they were.


---

## 5. Redoing an analysis

| To | Run |
|---|---|
| Analyse again, keeping the saved trap setup | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch --only Experiment4 --redo`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --only Experiment4 --redo" --wdir` |
| Redo the whole setup (boxes, entrances, thresholds) and analyse again | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch --only Experiment4 --redo --fresh`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --only Experiment4 --redo --fresh" --wdir` |
| Redo the entrance and thresholds of single traps, then analyse again | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch --only Experiment4 --traps 7,12`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --only Experiment4 --traps 7,12" --wdir` |
| Redo only the thresholds of one experiment (boxes and entrances kept) | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch --only Experiment4 --thresholds`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --only Experiment4 --thresholds" --wdir` |
| Redo only the thresholds of single traps | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch --only Experiment4 --traps 7 --thresholds`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --only Experiment4 --traps 7 --thresholds" --wdir` |
| Redo only the thresholds of every experiment | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch --thresholds`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --thresholds" --wdir` |
| Re-analyse everything | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch --redo`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --redo" --wdir` |

- Without `--only`, `--redo`, `--fresh` and `--thresholds` apply to every experiment in the
  config's folders.
- `--traps` needs exactly one experiment, so always pair it with `--only`. It keeps the trap box;
  to change a box or add a trap, use `--fresh`.
- `--traps` and `--thresholds` re-analyse by themselves; no `--redo` needed.

### The saved setup (session_cache.json)

Everything you chose in the setup windows is written to
`Output/<setup>/<experiment>/Run_Info/session_cache.json`. The next run of the same experiment
finds the file, skips the setup windows and goes straight to the analysis. (Results from older
versions kept it directly in the experiment folder; it is still found there and moved.)

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

Only these choices are stored. Every other setting is read from your config and
`config/pipeline/defaults.yaml` on each run, so you can change e.g. `scale_factor` or rupture
settings and re-run without repeating the setup.

- **Edit by hand:** the file is plain text. You can change a threshold, the pipette position or
  the smoothing of one trap, or delete a trap's entry to leave it out. Keep the JSON valid
  (commas, brackets). Deleting the file is the same as `--fresh`.
- **Thresholds and settings must match.** A threshold is only meaningful on the scale it was
  picked on. If `threshold_scale` doesn't match the current mask settings (`membrane_enhancement`,
  `membrane_snr_full_scale`), the file is ignored, a warning says why, and the setup runs again.
  Caches from before the contrast enhancement have no `threshold_scale` and count as the old
  CLAHE method, so they also trigger a new setup.
- **Changed data folder:** if the images differ from `source_files`, the log warns but the setup
  is still reused; check the results, or use `--fresh`.

---

## 6. Pooling experiments (bulk analysis)

After the experiments are analysed:

Terminal (Windows, macOS, Linux; built-in terminal of VS Code, PyCharm, ...):

```bash
python run_bulk_analysis.py
```

Spyder (IPython console):

```
%runfile C:/GitHub/MFAE_Analysis/run_bulk_analysis.py --wdir
```

| To | Run |
|---|---|
| Statistics and figures per setup, aspiration experiments as controls | Terminal: `python run_bulk_analysis.py`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_bulk_analysis.py --wdir` |
| Same, without shared aspiration controls | Terminal: `python run_bulk_analysis.py --controls none`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_bulk_analysis.py --args "--controls none" --wdir` |
| Results stored somewhere other than `Output/` | Terminal: `python run_bulk_analysis.py --output-root D:/MFAE_Output`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_bulk_analysis.py --args "--output-root D:/MFAE_Output" --wdir` |

It reads the experiment results, groups the cells by condition (from the experiment names) and
writes one folder per run, named by date and time, with a `README.txt` that lists every file and
what it shows.

- **Each setup is analysed on its own, never pooled with another.** Every folder in `Output/` that
  holds experiment results (e.g. `Symmetric_Efield/`, `Asymmetric_Efield/`) is one setup and gets
  its own `Bulk_Analysis_results/` inside it; experiments directly in `Output/` count as one more
  setup.
- **Aspiration-only controls are shared.** The aspiration-only (0 V) experiments in
  `Output/Aspiration/` are added as the control group to every setup's analysis, so each setup's
  pulse-vs-aspiration figures use them. `Aspiration/` also gets its own bulk analysis
  (aspiration-only mechanics). Use `--controls FOLDER` for a different control folder, or
  `--controls none` for no shared controls. Without such a folder (e.g. you only run one setup),
  the log says so and each setup uses only its own 0 V experiments as controls.

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
(`mfae.bulk.figure_data.load_all(...)`).

### Cell tags

The bulk analysis takes what happened to each cell (ruptured, not viable, detection failed, ...)
from your tags in each experiment's `Run_Info/cell_annotations.csv`; see
[Tagging the cells](#tagging-the-cells-review). Excluded cells are only counted in the attrition
table (`0_Tables/attrition_per_experiment.csv`). An unknown tag is reported and counted as a
detection failure; a row marked `CHECK: analysed cell changed` is reported and its tag used as it is.

Tags written into file names the earlier way (`Trap_03_protrusion_trace_R.csv`, and older
`Trap_XX_protrusion_and_shape_<TAG>.csv`, `trap_XX_detection_full_<TAG>.csv`) still work: they
are copied into the table at the next analysis or review, and used when the table has no tag for
that trap.

The bulk analysis also reads older result layouts (`Protrusion Length Detection/...`,
`Dye Uptake/Uptake traces/`, ..., and the flat one) with their old file names
(`trap_XX_detection_full.csv`, `Trap_XX_Uptake_Data.csv`, ...).

---

## 7. Naming experiments

The pulse settings of a run and the condition groups of the bulk analysis are both read from the
experiment name (the folder name, or the `.czi` file name), so every experiment must follow this
pattern (details in [`docs/experiment_naming.txt`](docs/experiment_naming.txt)):

```
YYMMDD_CellType_Treatment_Chip_Experiment-<P>Pa-<V>V-<D><ms|us|s>-frame<N>
260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-5ms-frame10      electroporated
260326_MDAMB231_WT_Chip2_Experiment283-1100Pa-0V-0ms-frame0       aspiration only
```

`…-100V-1ms-frame10` means a 100 V, 1 ms pulse at file number 10. Any voltage or duration works;
you never type them into the config. `0V` means aspiration only.

`--check` shows what it read from the name and warns when a name does not match the rule or its
pressure differs from `constant_pressure`. If a name can't follow the rule, set
`dye_uptake_parameters.has_pulse` and `pulse_frame` in your config instead; the bulk analysis will
skip that experiment.

---

## 8. Troubleshooting

| Problem or message | What to do |
|---|---|
| `SyntaxError: invalid syntax` when running Terminal: `python run_experiment.py ...`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "..." --wdir` | It is a terminal command but was typed into a Python console. Run it in a terminal from the repository folder, or in Spyder use the `%runfile` form (see [How to run a command](#how-to-run-a-command)). |
| Pressing F5 only prints the help text | No config was given; see [How to run a command](#how-to-run-a-command). |
| `... has N problem(s): - section.key: ...` | The named setting is invalid; the message says why. |
| `unknown setting (typo, or not supported ...)` | The key is misspelled or not a real setting. Compare with `config/pipeline/defaults.yaml`. |
| `Config key '...' is ignored: ...` | The setting exists in older configs but has no effect; you can delete it. |
| `is not valid YAML` | Usually inconsistent indentation or an unclosed quote. Windows paths are safest in single quotes. |
| `No .tif files found` | Check `paths.data_folder`. |
| `Dye analysis is enabled but no file names contain 'C2'` | Set `dye_channel_pattern` to the text that identifies your dye images. |
| The setup windows don't appear on a re-run | A saved setup is being reused; add `--fresh` (or `--traps` / `--thresholds` for parts of it). |
| Windows are too small on a high-resolution screen | Raise `workflow_settings.window_scale_factor` (e.g. 1.5) in `config/pipeline/defaults.yaml`. |
| The computer becomes sluggish | Lower `workflow_settings.n_workers` in your config. |
| A faint cell is barely captured in the threshold window | Press `F` for more smoothing, then lower the threshold (stay above ~3 sigma). |
| The mask includes background noise | Raise the threshold, or press `F` to reduce smoothing if it is high. |
| The protrusion mask reaches the left edge of the box | The protrusion is longer than the trap box, so its true length can't be measured; tracking stops after 3 frames at the edge. Draw a wider ROI (`--fresh`). |
| `Entry not confirmed for this trap` in the log | No valid protrusion was found for 3 frames in a row, so rupture detection is off for that trap. Watch its tracking movie. If the protrusion is tracked fine but longer than 0.9 × `channel_length_um` (36 µm by default), check `scale_factor`: `--check` shows the file's pixel size next to it. |
| A faint cell's length still flickers | Raise `membrane_temporal_window` (masks from the median of more frames) in `config/pipeline/defaults.yaml`, or try `--tip-tracking kymograph`. |
| A tracking movie won't play | Open it in VLC, which plays MJPG `.avi` files on any system. |

---

## 9. Reference

### 9.1 All commands

The asymmetric config is used as the example; swap in `config/config_symmetric.yaml` for the
symmetric microscope.

**Analyse**

| To | Run |
|---|---|
| Analyse all new experiments (asymmetric) | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch" --wdir` |
| Analyse all new experiments (symmetric) | Terminal: `python run_experiment.py config/config_symmetric.yaml --batch`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_symmetric.yaml --batch" --wdir` |
| Check everything first (no windows, writes nothing) | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch --check`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --check" --wdir` |
| Analyse a folder that isn't listed in the config | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch E:/Data/MFAE/Aspiration`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch E:/Data/MFAE/Aspiration" --wdir` |
| Only experiments whose name contains a text | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch --only Experiment4`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --only Experiment4" --wdir` |
| One experiment, by its folder (or `.czi` file) | Terminal: `python run_experiment.py config/config_asymmetric.yaml --data-folder M:/tnw/bn/gk/NN/2_Data-Analysis/MFAE/Asymmetric_Efield/5ms_100V/260114_MDAMB231_WT_Chip1_Experiment4-1100Pa-100V-5ms-frame10`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --data-folder M:/tnw/bn/gk/NN/2_Data-Analysis/MFAE/Asymmetric_Efield/5ms_100V/260114_MDAMB231_WT_Chip1_Experiment4-1100Pa-100V-5ms-frame10" --wdir` |
| Analyse with the kymograph tip tracker (this run only) | Terminal: `python run_experiment.py config/config_symmetric.yaml --batch --redo --tip-tracking kymograph`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_symmetric.yaml --batch --redo --tip-tracking kymograph" --wdir` |
| Review the movies and tag the cells (no config or raw data needed) | Terminal: `python run_experiment.py --review --only Experiment4`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "--review --only Experiment4" --wdir` |

**Redo:** see [Redoing an analysis](#5-redoing-an-analysis). **Pool:** see
[Pooling experiments](#6-pooling-experiments-bulk-analysis).

**Settings, new setups, help**

| To | Run |
|---|---|
| Show every setting a run uses, and where it comes from | Terminal: `python run_experiment.py config/config_asymmetric.yaml --batch --only Experiment4 --show-config`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "config/config_asymmetric.yaml --batch --only Experiment4 --show-config" --wdir` |
| New config for data recorded with other settings | Terminal: `python run_experiment.py --init-config config/my_setup.yaml --data-folder D:/MFAE/260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-1ms-frame10`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "--init-config config/my_setup.yaml --data-folder D:/MFAE/260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-1ms-frame10" --wdir` |
| See how a `.czi` file is organised (channels, frames) | Terminal: `python tools/czi_info.py D:/MFAE/261119_MDAMB231_LifeAct_Chip1_Experiment2-1100Pa-100V-1ms-frame304.czi`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/tools/czi_info.py --args "D:/MFAE/261119_MDAMB231_LifeAct_Chip1_Experiment2-1100Pa-100V-1ms-frame304.czi" --wdir` |
| List all options | Terminal: `python run_experiment.py --help`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "--help" --wdir` |

### 9.2 Command-line options

| Option | What it does |
|---|---|
| `--batch [FOLDER ...]` | Analyse every experiment in the config's folders (or the folders given) |
| `--check` | Validate config and images, then exit |
| `--only TEXT` | With `--batch`: only experiments whose name contains TEXT |
| `--redo` | With `--batch`: also analyse experiments that already have results |
| `--fresh` | Ignore the saved trap setup and redo the setup windows |
| `--traps N[,N...]` | Redo the pipette/threshold setup of only these traps, keep the rest |
| `--thresholds` | Redo only the threshold windows (all saved traps, or `--traps`); boxes and entrances kept |
| `--tip-tracking mask\|kymograph` | How the protrusion tip is found, for this run only (the log line `Protrusion tip: ...` shows which was used) |
| `--data-folder PATH` | Use this image folder (or `.czi` file) instead of the one in the config |
| `--experiment-id NAME` | Name of the results folder (default: the data folder's name) |
| `--setup NAME` | With `--batch`: name of the setup results folder |
| `--output-dir PATH` | Write results somewhere other than `Output/` |
| `--show-config` | Print every setting the run would use, and which file set it |
| `--init-config FILE` | Create a new config from a template in `config/pipeline/` (or `--from OTHER.yaml`) |
| `--review` | Watch the tracking movies and tag each trap's cell; with `--only TEXT`, `--traps N[,N...]` (see [Tagging the cells](#tagging-the-cells-review)) |
| `--export-traps N[,N...]\|all` | Developer export for this run: these traps' images to `Run_Info/Trap_images/` (see [For developers](#10-for-developers)) |
| `--debug` | Show detailed log messages |

### 9.3 Configs and settings

Settings live in two kinds of file that **never share a setting**:

| File | Contains | Who edits it |
|---|---|---|
| Your configs in `config/` (`config_asymmetric.yaml`, `config_symmetric.yaml`, and any made from the templates) | The user settings below | You |
| `config/pipeline/defaults.yaml` | Pipeline settings: image processing, detection, fitting, figures. The same for everyone | Developers only |

A user config holds these sections, and nothing else:

| Section | Settings |
|---|---|
| Experiment type | `experiment_type: ep` (electroporation) or `asp` (aspiration only); must agree with the name (`asp` needs `0V`) |
| Experiment | data folder(s), experiment name, microscope scale (`scale_factor`, µm per pixel), pressure, frame interval, max traps |
| Analyses (True / False) | dye uptake, actin, rupture detection, length smoothing, outlier rejection, viscoelastic fitting, kymographs; plus the channel names |
| Interactive setup | which frame is shown during setup, starting trap-box size |
| Computer | number of CPU cores |

Every one of these must be present. The pipeline refuses to start if a user config is missing a
user setting or contains a pipeline setting (a typo also ends up here), or if
`config/pipeline/defaults.yaml` contains a user setting.

- **To see exactly what a run will use**, add `--show-config`. Every value that doesn't come from
  `config/pipeline/defaults.yaml` is marked with where it came from.
- **Every run saves its complete settings** to `Run_Info/effective_config.yaml`. That file is a
  *full config* (it lists every setting and has no `experiment_type`), so you can rerun an old
  analysis with exactly the same settings even after the defaults have changed. Old full-length
  config files work the same way.

**A new config, for data recorded with other settings** (another objective, pressure or channel
order):

Terminal (Windows, macOS, Linux; built-in terminal of VS Code, PyCharm, ...):

```bash
python run_experiment.py --init-config config/my_setup.yaml --data-folder D:/MFAE/260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-1ms-frame10
```

Spyder (IPython console):

```
%runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args "--init-config config/my_setup.yaml --data-folder D:/MFAE/260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-1ms-frame10" --wdir
```

This copies `config/pipeline/template_tiff.yaml` (or `template_czi.yaml` when `--data-folder` is a
`.czi` file) into `config/`, with the data folder filled in. The experiment name defaults to the
folder's (or file's) name, and `experiment_type` is set from its voltage. To start from one of your
own configs instead, add `--from config/<your config>.yaml`. Then open the file and check the
values, in particular `scale_factor` and the channel settings. Because `--data-folder` overrides
the config, one config can serve a whole series of experiments recorded with the same settings.

### 9.4 .czi files

The pipeline reads Zeiss `.czi` files directly (`config/config_symmetric.yaml`, or a config made
from `template_czi.yaml`). The experiment name is the file name without `.czi`, so name the file by
the [naming rule](#7-naming-experiments). A CZI config differs from a TIFF one in two places:

```yaml
paths:
  image_format: czi          # tiff for a folder of exported .tif files
  data_folder: 'D:\MFAE'    # the .czi file, or its folder

dye_uptake_parameters:
  membrane_channel_number: 2 # channel numbers as Fiji shows them (dye first)
  dye_channel_number: 1
actin_parameters:
  channel_number: 2
```

Channels are chosen by number because the order in the CZI file (dye = 1, membrane = 2) is not the
order of the exported TIFFs (`C1` = membrane). TIFF configs use name patterns instead; each
format's settings are refused in the other, so a TIFF config can't be applied to a CZI by mistake.

Frame numbers count from 1, as in Zen and Fiji, so `frame10` is the 10th frame. Frame times are
read from the file when it stores them; otherwise `frame_interval` is used. `--check` shows which,
the channel names, and the file's pixel size next to `scale_factor`. Only single-position,
single-plane time-lapses are supported. To see how a CZI file is organised, use `tools/czi_info.py`
(see [All commands](#91-all-commands)).

### 9.5 How the masks and the length are measured

All settings named here are in `config/pipeline/defaults.yaml` under `image_processing`, unless
stated otherwise.

#### How the masks are made

The membrane channel is used only to build two masks per frame: the **protrusion** and the
**cell body**. Because no intensity is measured from it, it is enhanced to make the cell outline
clear and consistent from frame to frame:

1. Each frame is replaced by the pixel-wise median of itself and its neighbours, so noise at the
   threshold doesn't make the tip jitter or a faint protrusion flicker in and out
   (`membrane_temporal_window`, 1 = off). The image is then smoothed (1 px across the channel;
   along the channel by the per-trap faint-cell level).
2. In every frame, the background level and its noise are measured from the background peak of
   the histogram.
3. Each pixel becomes "noise units above background", so one threshold means the same thing in
   every frame, regardless of bleaching, exposure, or a very bright protrusion tip.
4. A pixel must also be at least 30 % as bright (above background) as the brightest signal across
   the channel in its column (`membrane_relative_level`, 0 = off). Around a bright cell the
   out-of-focus halo is well above the noise but only 10–20 % of the cell's brightness, so without
   this the masks are wider and blobbier than the cell. Only 2 px either side along the channel
   count, so the dim shaft next to a bright spot in the protrusion is kept.
5. After thresholding, background enclosed by the cell is filled in (membrane stains are brightest
   at the rim, which would otherwise give ring-shaped masks), and small specks are removed.
6. If the protrusion mask runs on to the edge of the box through faint signal ahead of the tip
   (out-of-focus light or debris in the channel), it is cut where the signal along the channel
   stays below 40 % of the protrusion's own level (`protrusion_leak_fraction`). Masks that end
   inside the box are never cut.
7. The body is the largest blob at the entrance (specks there are ignored). With a faint cell the
   medium next to the trap, or a passing cell, can join the body in single frames; a body mask more
   than 1.3× as large as the body in the frames around it keeps only the part within 3 px of that
   body (`body_temporal_check`, `False` = off).

**Dye and actin intensities are always measured on the raw pixels.** They only borrow the masks;
this also holds when actin is imaged in the same channel as the membrane.

The enhancement settings are `membrane_enhancement`, `membrane_denoise_sigma`,
`membrane_denoise_sigma_along`, `membrane_temporal_window`, `membrane_snr_full_scale`,
`membrane_fill_holes`, `membrane_max_hole_fraction` and `membrane_relative_level`. Set
`membrane_enhancement: False` to return to the older CLAHE-based segmentation; thresholds picked in
one mode are not comparable with the other.

#### Where the tip is (protrusion length)

There are two ways to find the tip, set by `tip_tracking` (or `--tip-tracking` for one run). In
both, *when* a cell is there (entry, exit, a new cell arriving) is decided from the masks, and
areas and shape come from the masks.

**`mask` (default): from each frame's mask.** The tip is placed on the protrusion's centre-line
profile, where the intensity drops to 25 % of the way from the channel background ahead of the tip
to the protrusion just inside it (`tip_level_fraction`; lower follows dimmer tips). If the profile
is still above that level 15 px beyond the mask edge (signal in the channel, not the tip), or the
contrast is too low, the mask edge is used instead (`max_tip_refinement_px`).

**`kymograph`: followed through time.** The membrane profile along the channel centre line is
followed through time, as you would read the edge on the kymograph. In every frame each position
gets a score for looking like the protrusion's outer edge (protrusion signal on the entrance side,
nothing left on the channel side, judged in noise units so a faint far end counts and an internal
step next to it does not), and the tip path through all frames is the one with the best total
score, with a penalty for jumping between frames (`tip_jump_cost`). A wrong edge in single frames
or short runs can't pull the path away; bright edges are followed however fast they move. Where a
faint far end drops below the threshold the mask gets shorter, but the tracked length does not.
Where the protrusion end is clearly above the mask threshold (at least 1.5×), the tip is placed
where the centre-line profile drops to that threshold, so it sits on the mask edge. The tracker
only moves the tip inside the analysed cell, tracked over that cell's frames alone. The trace CSV
also has the mask-based length (`Protrusion_Length_mask_um`) for comparison.

**Length spikes (faint cells).** With a faint cell the tip can jump for a frame or two onto faint
signal ahead of it, or lose the faint far end of the protrusion. Each frame's length is compared
with the median of the 7 frames around it; a frame that is at least 2 µm and 3 robust standard
deviations away is replaced by that median. Runs of up to 3 frames count as spikes; anything that
lasts longer (slip, a new cell, real growth) is kept, at its exact frame. This runs before entry
and exit detection, so a spike can't end tracking. The trace CSV keeps the measured value
(`Protrusion_Length_raw_um`) and marks the replaced frames (`Spike_Removed`). Settings, under
`rupture_detection`: `spike_filter_window` (0 = off), `spike_filter_min_jump_um`,
`spike_filter_n_mad`.

**When a cell ends** (see [Several cells in one trap](#4-checking-the-results)): the settings are
`rupture_detection.exit_at_roi_edge_frames` (tip at the box edge) and
`rupture_detection.cell_change_drop_ratio` (length drop that means a new cell).

#### Figures and movies

- One PDF per trap in addition to the multipanel PDFs: `plotting_parameters.save_individual_trap_plots: True`;
  panels per row: `plotting_parameters.multipanel_columns`.
- Tracking movies: `workflow_settings.save_tracking_movies` (on by default), frame rate and size
  `workflow_settings.tracking_movie_fps` and `tracking_movie_width_px`.

---

## 10. For developers

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
| `mfae/config/loader.py` | Combining `config/pipeline/defaults.yaml` with a user config (`USER_KEYS` lists the user settings) |
| `mfae/config/naming.py` | The experiment-name rule (pulse settings, bulk condition groups) |
| `mfae/files/readers.py` | Finding images (.tif / .czi), channel matching, timestamps |
| `mfae/files/output_layout.py` | Folder layout and file names of the results (bulk readers use it too) |
| `mfae/files/trap_export.py` | Developer export of trap images (`--export-traps`) |
| `mfae/files/annotations.py`, `mfae/review.py` | Cell tags (`Run_Info/cell_annotations.csv`) and the `--review` window |
| `mfae/imaging/trap_rois.py` | Interactive rotation, ROI and trap selection, cropping |
| `mfae/membrane/detection.py` | Masks, protrusion length, cells per trap, entry/exit and rupture detection, threshold windows |
| `mfae/membrane/kymo_tip.py` | Tip tracking on the kymograph (`tip_tracking: kymograph`) |
| `mfae/membrane/kymograph.py`, `tracking_movie.py` | Membrane kymographs and per-trap tracking movies |
| `mfae/quantification/dye_uptake.py`, `actin.py` | Dye and actin channels, measured in the membrane masks |
| `mfae/mechanics/models.py`, `fitting.py` | Channel geometry and flow, viscoelastic models, fitting |
| `mfae/plotting/figures.py`, `multipanel.py` | Per-experiment figures (single-trap and all-trap) |
| `mfae/utils.py` | Shared helpers, including the membrane enhancement and mask generation |
| `mfae/bulk/` | Pooling experiments: `run.py` (pipeline), `loader.py`, `mechanics.py`, `layout.py`, `figure_data.py`, `plots/` |

Import modules by their package path (`from mfae.membrane import detection`); the scripts and
tests put the project folder on the path. To change where results are written, edit
`mfae/files/output_layout.py` only; the pipeline and the bulk analysis both read it. The bulk
result layout is defined in `mfae/bulk/layout.py`.

**Exporting trap images to work on a dataset elsewhere.** A raw `.czi` is often too large to
share. The developer export writes each chosen trap's frames exactly as the analysis works on them
(rotated, cropped to the trap box, original bit depth, lossless), together with the trap's setup
(entrance, thresholds, smoothing), the frame times and the pixel size, to
`Run_Info/Trap_images/Trap_XX_images.npz`: about 4 MB per trap for the membrane channel, plus the
dye and actin channels when those analyses are on. It is off by default.

| To | Do |
|---|---|
| Export for one run | Add `--export-traps 1,5,6` (or `--export-traps all`) to the command |
| Export for every run | In `config/pipeline/defaults.yaml`, under `workflow_settings`: `export_trap_images: True`, and `export_trap_images_traps: [1, 5, 6]` (`[]` = every analysed trap) |
| Export from an experiment that was already analysed, without analysing again | Terminal: `python tools/export_trap_stacks.py Experiment5 --traps 1,5,6`<br>Spyder: `%runfile C:/GitHub/MFAE_Analysis/tools/export_trap_stacks.py --args "Experiment5 --traps 1,5,6" --wdir` |

The file is written before the detection runs, so it exists even when the detection fails for that
trap. The log says when the export is on. Read a file back with
`arrays, meta = mfae.files.trap_export.read(path)`: `arrays["membrane"]` is (frames, height, width),
and `meta` holds the setup and settings needed to rerun the detection on it.

When you add a config setting, define it in `mfae/config/schema.py` (unknown keys are rejected, so
typos can't be silently ignored). Then decide who owns it: a pipeline setting goes, with its value
and an explanation, in `config/pipeline/defaults.yaml`; a user setting goes in `USER_KEYS` in
`mfae/config/loader.py` and in every user config: the two templates in `config/pipeline/` and
`config/config_asymmetric.yaml`, `config/config_symmetric.yaml`.
