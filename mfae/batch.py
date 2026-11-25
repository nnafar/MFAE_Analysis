# -*- coding: utf-8 -*-
"""
Batch runs: analyse every experiment found under one folder, with one config.

    python run_experiment.py config/config_czi_symmetric.yaml --batch "M:/.../Symmetric_Efield"

* Every experiment under the folder (at any depth) is found: each .czi file
  for a CZI config, each folder holding .tif files for a TIFF config. Only
  names that follow the experiment-naming rule are taken; the pulse and
  ep / asp (0 V = aspiration only) come from each name.
* One setup = one folder + one config. Results go to Output/<setup>/<experiment>/,
  <setup> being the folder's name (or --setup NAME), so experiments of
  different setups (e.g. symmetric and asymmetric field) are never mixed;
  the bulk analysis also treats each setup on its own.
* Pass 1 does the interactive setup of every experiment, one after another
  (saved setups are reused; Esc stops the batch and keeps the setups done so
  far). Pass 2 analyses them all without interaction, e.g. overnight.
* Experiments that already have results are skipped, so a batch can be run
  again after new data were added; --redo analyses them again.
* A failure in one experiment is reported and the batch carries on. A summary
  is printed and saved as Output/<setup>/batch_summary_<date>_<time>.txt.
"""

import logging
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


def batch_folder_and_format(config_path: Path, folder: Optional[str]) -> Tuple[Path, str]:
    """The batch folder (given, or the config's data_folder) and the config's image format."""
    raw = config_loader.read_yaml(Path(config_path))
    paths = raw.get("paths") or {}
    root = Path(folder) if folder else Path(str(paths.get("data_folder", "")))
    if root.suffix.lower() == ".czi":
        root = root.parent                 # a single .czi file given: use its folder
    return root, str(paths.get("image_format", "tiff")).lower()


def run_batch(args) -> int:
    """Run a batch (see the module docstring). Returns a process exit code."""
    from mfae.cli import check_setup, load_config, save_effective_config, show_config

    try:
        root, image_format = batch_folder_and_format(args.config_path, args.batch)
        found = find_experiments(root, image_format)
    except (FileNotFoundError, config_loader.ConfigFileError, OSError) as e:
        logger.error(str(e))
        return 1

    setup = args.setup or root.name
    out_root = Path(args.output_dir or default_output_root()) / setup
    what = ".czi files" if image_format == "czi" else "folders with .tif files"
    logger.info(f"Batch '{setup}': {len(found.experiments)} experiment(s) ({what}) under {root}")
    logger.info(f"Results go to {out_root}")
    for p in found.badly_named:
        logger.warning(f"Skipped (name doesn't follow the naming rule): {p}")
    if found.other_format:
        other = "TIFF folders" if image_format == "czi" else ".czi files"
        logger.warning(f"Also found {len(found.other_format)} {other}; they need a config with "
                       f"image_format: {'tiff' if image_format == 'czi' else 'czi'} and are not "
                       f"analysed in this batch.")
    if not found.experiments:
        logger.error(f"No {what} with a valid experiment name under {root}.")
        return 1
    if args.experiment_id:
        logger.warning("--experiment-id is ignored with --batch: each experiment is named after "
                       "its file or folder.")

    status: Dict[str, str] = {}
    configs = {}
    for exp_path in found.experiments:
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
        configs[name] = (config, layered, exp_path)

    if args.show_config:
        for name, (config, layered, _) in configs.items():
            print(f"===== {name} =====\n{show_config(config, layered)}")
        return 0

    if args.check:
        for name, (config, _, _) in configs.items():
            logger.info(f"\n===== Check: {name} =====")
            status[name] = "check passed" if check_setup(config, out_root) else "check: PROBLEM"
        _summary(status, setup, root, out_root, save=False)
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
    for k, (name, (config, layered, exp_path)) in enumerate(configs.items(), 1):
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
        config, layered, exp_path = configs[name]
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
    _summary(status, setup, root, out_root, save=True)
    ok = ("done", "already analysed")
    return 0 if all(v.startswith(ok) for v in status.values()) else 1


def _summary(status: Dict[str, str], setup: str, root: Path, out_root: Path, save: bool) -> None:
    lines = [f"  {name}: {state}" for name, state in status.items()]
    counts: Dict[str, int] = {}
    for state in status.values():
        key = state.split(" (")[0].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    text = (f"== BATCH SUMMARY: {setup} ==\n" + "\n".join(lines) + "\n"
            + ", ".join(f"{n} {k}" for k, n in counts.items()))
    logger.info("\n" + text)
    if not save:
        return
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        path = out_root / f"batch_summary_{datetime.now():%Y%m%d_%H%M}.txt"
        path.write_text(f"Batch over {root}\nResults in {out_root}\n\n{text}\n", encoding="utf-8")
        logger.info(f"Batch summary saved to {path}")
    except OSError:
        logger.warning("Could not write the batch summary file.", exc_info=True)
