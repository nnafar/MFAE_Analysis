# -*- coding: utf-8 -*-
"""
Batch runs: analyse every experiment in the folders you choose, with one config.

    python run_experiment.py config/config_asymmetric_tiff.yaml --batch
    python run_experiment.py config/config_asymmetric_tiff.yaml --batch "M:/.../Asymmetric_Efield/5ms_100V"

* Folders: the ones given after --batch, else the config's data_folder (one
  folder or a list). Experiments with other settings go in another config
  with their own folders, run the same way.
* Every experiment in those folders (at any depth) is found: each .czi file
  for a CZI config, each folder holding .tif files for a TIFF config. Only
  names that follow the experiment-naming rule are taken; the pulse and
  ep / asp (0 V = aspiration only) come from each name.
* Results go to Output/<setup>/<experiment>/. <setup> is the folder's name
  (Symmetric_Efield, Aspiration) or, for a condition folder such as 5ms_100V,
  the setup folder holding it (Asymmetric_Efield); --setup NAME sets it. So
  different setups are never mixed, and configs for parts of one setup
  still fill the same setup folder.
* Pass 1 does the interactive setup of every experiment, one after another
  (saved setups are reused; Esc stops the batch and keeps the setups done so
  far). Pass 2 analyses them all without interaction, e.g. overnight.
* Experiments that already have results are skipped, so a batch can be run
  again after new data were added; --redo analyses them again.
* A failure in one experiment is reported and the batch carries on. A summary
  is printed and saved as Output/<setup>/batch_summary_<date>_<time>.txt.
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mfae import utils
from mfae.config import loader as config_loader
from mfae.config import naming as experiment_naming
from mfae.files import output_layout
from mfae.pipeline import MFAAnalysis, default_experiment_id, default_output_root

logger = logging.getLogger(__name__)

TIF_SUFFIXES = (".tif", ".tiff")


@dataclass
class Found:
    """What a scan of the batch folder found."""
    experiments: List[Path]                 # matching the config's image format, well named
    badly_named: List[Path]                 # right format, but the name breaks the naming rule
    other_format: List[Path]                # experiments of the other image format (not analysed)


def _has_tifs(folder: Path) -> bool:
    try:
        return any(f.suffix.lower() in TIF_SUFFIXES for f in folder.iterdir() if f.is_file())
    except OSError:
        return False


def find_experiments(root: Path, image_format: str) -> Found:
    """
    Every experiment under `root`, at any depth: .czi files (image_format
    'czi') or folders that directly hold .tif files ('tiff'). Folders inside
    an MFAE results folder (Output/...) are ignored.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Batch folder not found (or not a folder): {root}")
    czis, tif_dirs = [], []
    for p in sorted([root] + list(root.rglob("*"))):
        if "Bulk_Analysis_results" in p.parts or output_layout.RUN_INFO in p.parts:
            continue
        if p.is_file() and p.suffix.lower() == ".czi":
            czis.append(p)
        elif p.is_dir() and _has_tifs(p):
            tif_dirs.append(p)
    mine, other = (czis, tif_dirs) if image_format == "czi" else (tif_dirs, czis)
    good, bad = [], []
    for p in mine:
        (good if experiment_naming.parse_experiment_name(default_experiment_id(p)) else bad).append(p)
    return Found(good, bad, other)


def is_analysed(results_dir: Path) -> bool:
    """True when an experiment's results folder holds a finished analysis."""
    return output_layout.experiment_file(output_layout.build(results_dir), 'trap_summary').exists()


CONDITION_FOLDER = re.compile(r"^\d+(?:\.\d+)?(?:us|ms|s)_\d+V$", re.IGNORECASE)


def setup_name(folder: Path) -> str:
    """
    Results sub-folder (setup) of a batch folder: the folder's own name
    (Symmetric_Efield, Aspiration), or, for a condition folder such as
    5ms_100V or 100us_100V, the name of the setup folder that holds it.
    """
    folder = Path(folder)
    if CONDITION_FOLDER.match(folder.name) and folder.parent.name:
        return folder.parent.name
    return folder.name


def setup_for_experiment(experiment: Path, folders: List[Path]) -> str:
    """
    Setup (results sub-folder) of one experiment run on its own with a config
    that lists folders: the same as a batch over those folders would use, so
    single runs and batch runs fill the same Output/<setup>/.
    """
    exp = os.path.normcase(os.path.abspath(str(experiment)))
    for folder in folders:
        f = os.path.normcase(os.path.abspath(str(folder))).rstrip("\\/")
        if exp == f or exp.startswith(f + os.sep):
            return setup_name(Path(folder))
    return setup_name(Path(experiment).parent)


def batch_folders_and_format(config_path: Path, folders: Optional[List[str]]) -> Tuple[List[Path], str]:
    """
    The folders to search (given after --batch, else the config's data_folder,
    which may be one folder or a list) and the config's image format.
    """
    raw = config_loader.read_yaml(Path(config_path))
    paths = raw.get("paths") or {}
    chosen = list(folders or [])
    if not chosen:
        df = paths.get("data_folder", "")
        chosen = [str(x) for x in df] if isinstance(df, list) else [str(df)]
    roots = []
    for f in chosen:
        root = Path(f)
        if root.suffix.lower() == ".czi":
            root = root.parent             # a single .czi file given: use its folder
        roots.append(root)
    return roots, str(paths.get("image_format", "tiff")).lower()


def run_batch(args) -> int:
    """Run a batch (see the module docstring). Returns a process exit code."""
    from mfae.cli import check_setup, load_config, save_effective_config, show_config

    try:
        roots, image_format = batch_folders_and_format(args.config_path, args.batch)
    except (FileNotFoundError, config_loader.ConfigFileError, OSError) as e:
        logger.error(str(e))
        return 1

    what = ".czi files" if image_format == "czi" else "folders with .tif files"
    out_base = Path(args.output_dir or default_output_root())
    selected: List[Tuple[Path, Path]] = []          # (experiment, results root of its setup)
    for root in roots:
        try:
            found = find_experiments(root, image_format)
        except (FileNotFoundError, OSError) as e:
            logger.error(str(e))
            return 1
        setup = args.setup or setup_name(root)
        logger.info(f"{root}: {len(found.experiments)} experiment(s) ({what}) -> Output/{setup}/")
        for p in found.badly_named:
            logger.warning(f"Skipped (name doesn't follow the naming rule): {p}")
        if found.other_format:
            other = "TIFF folders" if image_format == "czi" else ".czi files"
            logger.warning(f"Also found {len(found.other_format)} {other} in {root}; they need a "
                           f"config with image_format: {'tiff' if image_format == 'czi' else 'czi'} "
                           f"and are not analysed in this batch.")
        selected += [(e, out_base / setup) for e in found.experiments]
    if not selected:
        logger.error(f"No {what} with a valid experiment name in: " + ", ".join(map(str, roots)))
        return 1
    if args.experiment_id:
        logger.warning("--experiment-id is ignored with --batch: each experiment is named after "
                       "its file or folder.")

    status: Dict[str, str] = {}
    configs = {}
    for exp_path, out_root in selected:
        name = default_experiment_id(exp_path)
        if name in configs:
            status[name + " (duplicate)"] = f"skipped: same name as an earlier experiment ({exp_path})"
            continue
        if not args.redo and not args.check and not args.show_config and is_analysed(out_root / name):
            status[name] = "already analysed (skipped; --redo to analyse again)"
            continue
        config, layered = load_config(args.config_path, exp_path, type_from_name=True)
        if config is None:
            status[name] = "skipped: config problem (see above)"
            continue
        if args.debug:
            config.workflow_settings.debug_mode = True
        configs[name] = (config, layered, exp_path, out_root)

    if args.show_config:
        for name, (config, layered, _, _) in configs.items():
            print(f"===== {name} =====\n{show_config(config, layered)}")
        return 0

    if args.check:
        for name, (config, _, _, out_root) in configs.items():
            logger.info(f"\n===== Check: {name} =====")
            status[name] = "check passed" if check_setup(config, out_root) else "check: PROBLEM"
        _summary(status, roots, set(), save=False)
        return 0 if all(v == "check passed" for v in status.values()
                        if not v.startswith("already")) else 1

    if image_format == "czi":
        try:
            import pylibCZIrw  # noqa: F401  (only checks it is installed)
        except ImportError:
            logger.error("Reading .czi files needs the pylibCZIrw package, which isn't installed. "
                         "Install it (in Spyder: %pip install pylibCZIrw czifile, then restart "
                         "the kernel; in a terminal: pip install pylibCZIrw czifile).")
            return 1

    # --- Pass 1: interactive setups (or the saved ones) ---------------------
    logger.info(f"\n== BATCH PASS 1: SETUP OF {len(configs)} EXPERIMENT(S) ==")
    ready: List[str] = []
    for k, (name, (config, layered, exp_path, out_root)) in enumerate(configs.items(), 1):
        logger.info(f"\n--- Setup {k}/{len(configs)}: {name} ---")
        analysis = None
        try:
            analysis = MFAAnalysis(exp_path, name, config, output_root=out_root,
                                   use_session_cache=not args.fresh)
            save_effective_config(config, layered, args.config_path, analysis.results_dir)
            if analysis.run_analysis(setup_only=True):
                ready.append(name)
            elif analysis.stopped_by_user:
                status[name] = "stopped (Esc during setup)"
                logger.warning("Batch stopped during setup (Esc). The setups finished so far are "
                               "saved; run the batch again to continue.")
                break
            else:
                status[name] = "failed during setup (see its Run_Info/analysis.log)"
        except KeyboardInterrupt:
            logger.warning("Batch interrupted by user.")
            return 130
        except Exception:
            status[name] = "failed during setup (see its Run_Info/analysis.log)"
            logger.error(f"Setup of {name} failed.", exc_info=True)
        finally:
            if analysis:
                analysis.cleanup()

    # --- Pass 2: analyses, no interaction -----------------------------------
    logger.info(f"\n== BATCH PASS 2: ANALYSIS OF {len(ready)} EXPERIMENT(S) ==")
    for k, name in enumerate(ready, 1):
        config, layered, exp_path, out_root = configs[name]
        logger.info(f"\n--- Analysis {k}/{len(ready)}: {name} ---")
        analysis = None
        try:
            analysis = MFAAnalysis(exp_path, name, config, output_root=out_root,
                                   use_session_cache=True, log_mode='a')
            status[name] = "done" if analysis.run_analysis() else "failed (see its Run_Info/analysis.log)"
        except KeyboardInterrupt:
            logger.warning("Batch interrupted by user.")
            status[name] = "interrupted"
            break
        except Exception:
            status[name] = "failed (see its Run_Info/analysis.log)"
            logger.error(f"Analysis of {name} failed.", exc_info=True)
        finally:
            if analysis:
                analysis.cleanup()

    utils.setup_logging(debug_mode=args.debug, log_file=None)   # summary: console only
    _summary(status, roots, {o for *_, o in configs.values()}, save=True)
    ok = ("done", "already analysed")
    return 0 if all(v.startswith(ok) for v in status.values()) else 1


def _summary(status: Dict[str, str], roots: List[Path], out_roots: set, save: bool) -> None:
    lines = [f"  {name}: {state}" for name, state in status.items()]
    counts: Dict[str, int] = {}
    for state in status.values():
        key = state.split(" (")[0].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    text = ("== BATCH SUMMARY ==\n" + "\n".join(lines) + "\n"
            + ", ".join(f"{n} {k}" for k, n in counts.items()))
    logger.info("\n" + text)
    if not save:
        return
    header = "Batch over:\n" + "\n".join(f"  {r}" for r in roots) + "\n\n"
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    for out_root in sorted(out_roots):
        try:
            out_root.mkdir(parents=True, exist_ok=True)
            path = out_root / f"batch_summary_{stamp}.txt"
            path.write_text(header + text + "\n", encoding="utf-8")
            logger.info(f"Batch summary saved to {path}")
        except OSError:
            logger.warning("Could not write the batch summary file.", exc_info=True)
