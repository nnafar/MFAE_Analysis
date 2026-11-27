# -*- coding: utf-8 -*-
"""
Command line of run_experiment.py: analyse one experiment.

    python run_experiment.py my_exp.yaml                  interactive setup, then analysis
    python run_experiment.py my_exp.yaml --check          check config and images, no windows
    python run_experiment.py my_exp.yaml --show-config    every setting and where it comes from
    python run_experiment.py --init-config new.yaml --data-folder D:/data/<experiment>

The analysis itself is in mfae.pipeline (MFAAnalysis).
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from mfae import utils
from mfae.config import loader as config_loader
from mfae.config import naming as experiment_naming
from mfae.config.schema import MFAConfig, validate_config, format_validation_error
from mfae.files import output_layout
from mfae.files.readers import make_file_reader
from mfae.paths import PROJECT_ROOT
from mfae.pipeline import (DEFAULT_CONFIG, DEFAULT_CZI_CONFIG, MFAAnalysis,
                           default_experiment_id, default_output_root, threshold_scale)

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run_experiment.py",
        description="Microfluidic aspiration (MFA) analysis of one experiment folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical workflow:\n"
            "  1. python run_experiment.py --init-config my_exp.yaml --data-folder D:/data/260128_...\n"
            "  2. python run_experiment.py my_exp.yaml --check     (no windows; checks config and images)\n"
            "  3. python run_experiment.py my_exp.yaml             (interactive setup, then analysis)\n"
            "\n"
            "Many experiments at once (one config per setup):\n"
            "  python run_experiment.py config/my_setup.yaml --batch                (folders listed in the config)\n"
            "  python run_experiment.py config/my_setup.yaml --batch D:/data/A D:/data/B\n"
            "\n"
            "Your file holds the experiment's settings; everything else comes from\n"
            "config/pipeline/defaults.yaml. See README.md for details."
        ),
    )
    parser.add_argument("config_path", nargs="?", type=Path,
                        help="Path to the YAML config file.")
    parser.add_argument("--data-folder", type=Path,
                        help="Override paths.data_folder (folder with the .tif images).")
    parser.add_argument("--experiment-id",
                        help="Override paths.experiment_id (name of the results folder). "
                             "Defaults to the data folder's name when --data-folder is given.")
    parser.add_argument("--output-dir", type=Path,
                        help=f"Where result folders are written (default: {default_output_root()}).")
    parser.add_argument("--check", action="store_true",
                        help="Validate the config and scan the images, then exit. Opens no windows.")
    parser.add_argument("--batch", nargs="*", default=None, metavar="FOLDER",
                        help="Analyse every experiment in the given folder(s), at any depth: each "
                             ".czi file (CZI config) or each folder of .tif files (TIFF config). "
                             "Without folders: the config's data_folder (one folder or a list). "
                             "All setups first, then all analyses. Results go to "
                             "<output-dir>/<setup>/, the setup being the folder's name, or its "
                             "parent's for a condition folder such as 5ms_100V.")
    parser.add_argument("--setup", metavar="NAME",
                        help="With --batch: name of the results sub-folder (setup) for all "
                             "folders (default: worked out per folder, e.g. Asymmetric_Efield).")
    parser.add_argument("--redo", action="store_true",
                        help="With --batch: also analyse experiments that already have results.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore a saved session_cache.json and redo the interactive trap setup.")
    parser.add_argument("--traps", type=_trap_list, metavar="N[,N...]",
                        help="Redo the pipette/threshold setup of only these traps (numbers as in "
                             "the movies and plots, e.g. 7 or 7,12); all other traps keep the saved "
                             "setup. The experiment is then analysed again. With --batch, pick the "
                             "experiment with --only.")
    parser.add_argument("--only", metavar="TEXT",
                        help="With --batch: only experiments whose name contains TEXT "
                             "(e.g. Experiment4).")
    parser.add_argument("--debug", action="store_true",
                        help="Show detailed (DEBUG) log messages on screen.")
    parser.add_argument("--show-config", action="store_true",
                        help="Print every setting this run would use, and which file set it, then exit.")
    parser.add_argument("--init-config", type=Path, metavar="NEW_CONFIG",
                        help="Create a new config by copying an existing one with the new "
                             "--data-folder / --experiment-id filled in, then exit.")
    parser.add_argument("--from", dest="copy_from", type=Path, default=None,
                        metavar="CONFIG",
                        help=f"Config copied by --init-config (default: {DEFAULT_CONFIG.name}, "
                             f"or {DEFAULT_CZI_CONFIG.name} when --data-folder is a .czi file).")
    return parser


def _trap_list(text: str) -> List[int]:
    """'7' or '7,12' -> [7, 12] (trap numbers as shown in the movies, from 1)."""
    try:
        traps = sorted({int(t) for t in str(text).replace(" ", "").split(",") if t})
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected trap numbers like 7 or 7,12, got '{text}'")
    if not traps or min(traps) < 1:
        raise argparse.ArgumentTypeError(f"trap numbers start at 1, got '{text}'")
    return traps


def _set_yaml_value(text: str, key: str, value: str) -> str:
    """
    Replace the value of the first `key: ...` line, keeping comments and layout.
    A list value ('- ...' lines below the key, e.g. several data folders,
    including switched-off '# - ...' lines) is replaced as a whole.
    """
    import re
    quoted = value if key == "experiment_type" else "'" + str(value).replace("'", "''") + "'"
    pattern = re.compile(rf"^(?P<ind>[ \t]*){key}:[^\n]*(?P<items>(?:\n(?P=ind)[ \t]+#?[ \t]*-[^\n]*)*)",
                         re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"no '{key}:' entry to fill in")
    return pattern.sub(lambda m: f"{m.group('ind')}{key}: {quoted}", text, count=1)


def init_config(dest: Path, source: Optional[Path], data_folder: Optional[Path],
                experiment_id: Optional[str]) -> bool:
    """
    Copy an existing config (by default config/pipeline/template_tiff.yaml, or
    config/pipeline/template_czi.yaml for a .czi file) to `dest`, replacing its
    data folder and experiment name, and setting experiment_type from the
    voltage in the new name (0 V = asp). All other settings are kept, so
    copying the config of a similar experiment is the quickest way to start.
    """
    if dest.exists():
        logger.error(f"{dest} already exists; choose another name or delete it first.")
        return False
    is_czi = data_folder is not None and Path(data_folder).suffix.lower() == ".czi"
    if source is None:
        source = DEFAULT_CZI_CONFIG if is_czi else DEFAULT_CONFIG
    if not source.exists():
        logger.error(f"Config to copy not found: {source}")
        return False

    text = source.read_text(encoding="utf-8")
    if is_czi and "image_format: czi" not in text:
        logger.error(f"{data_folder} is a .czi file but {source.name} is a TIFF config. "
                     f"Start from {DEFAULT_CZI_CONFIG.relative_to(PROJECT_ROOT)} instead "
                     f"(--from), or from another CZI config.")
        return False
    if data_folder is not None:
        experiment_id = experiment_id or default_experiment_id(data_folder)
    try:
        if data_folder is not None:
            text = _set_yaml_value(text, "data_folder", str(data_folder))
        if experiment_id:
            text = _set_yaml_value(text, "experiment_id", experiment_id)
            name = experiment_naming.parse_experiment_name(experiment_id)
            if name is not None and "experiment_type:" in text:
                text = _set_yaml_value(text, "experiment_type", "ep" if name.has_pulse else "asp")
    except ValueError as e:
        logger.error(f"Cannot fill in {source.name}: {e}.")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    logger.info(f"Created {dest} (copy of {source.name}).")
    if data_folder is None and not experiment_id:
        logger.info("Next: open it and set paths.data_folder and paths.experiment_id.")
    logger.info(f"Then check it with:  python run_experiment.py {dest} --check")
    return True


def load_config(config_path: Path, data_folder: Optional[Path] = None,
                experiment_id: Optional[str] = None, type_from_name: bool = False
                ) -> Tuple[Optional[MFAConfig], Optional[config_loader.LayeredConfig]]:
    """
    Combine the defaults, the user's file and command-line overrides, then
    validate. Logs a readable error and returns (None, None) on failure.
    type_from_name: take experiment_type (ep / asp) from the experiment name
    instead of the config (batch runs over folders with mixed experiments).
    """
    overrides: Dict[str, Any] = {}
    if data_folder is not None:
        overrides["data_folder"] = str(data_folder)
        experiment_id = experiment_id or default_experiment_id(data_folder)
    if experiment_id is not None:
        overrides["experiment_id"] = experiment_id

    type_override = None
    if type_from_name and experiment_id is not None:
        info = experiment_naming.parse_experiment_name(experiment_id)
        if info is not None:
            type_override = "ep" if info.has_pulse else "asp"
    try:
        layered = config_loader.load_layers(config_path, {"paths": overrides} if overrides else None,
                                            experiment_type_override=type_override)
    except config_loader.ConfigFileError as e:
        logger.error(str(e))
        return None, None

    try:
        return validate_config(layered.data), layered
    except ValidationError as e:
        logger.error(f"{config_path} has {e.error_count()} problem(s):\n"
                     f"{format_validation_error(e, layered.sources)}")
    except ValueError as e:
        logger.error(f"{config_path}: {e}")
    return None, None


def show_config(config: MFAConfig, layered: config_loader.LayeredConfig) -> str:
    """All settings of a run, each value annotated with the file that set it (blank = defaults)."""
    header = (f"# Settings from: {' <- '.join(layered.layers)}\n"
              f"# Values without a comment come from config/pipeline/defaults.yaml.\n\n")
    return header + config_loader.render_settings(
        config.model_dump(mode="json"), layered.sources, hide_source="defaults")


def save_effective_config(config: MFAConfig, layered: config_loader.LayeredConfig,
                          config_path: Path, results_dir: Path) -> Path:
    """Record exactly which settings produced this run's results."""
    from datetime import datetime
    path = output_layout.experiment_file(output_layout.build(results_dir), 'effective_config')
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Complete settings used for this run (written {datetime.now():%Y-%m-%d %H:%M}).\n"
              f"# Config file: {Path(config_path).resolve()}\n"
              f"# This file is a full config: run it again with\n"
              f"#     python run_experiment.py {path.name}\n")
    path.write_text(header + show_config(config, layered).split("\n", 2)[2], encoding="utf-8")
    return path


def check_setup(config: MFAConfig, output_root: Optional[Path] = None) -> bool:
    """
    Dry run for new users: scans the image folder with the same code the
    pipeline uses and reports what it found, without opening any windows
    or writing results. Returns False if a problem would stop the analysis.
    """
    params = config.model_dump()
    problems: List[str] = []
    notes: List[str] = []

    reader = make_file_reader(config.paths.data_folder, params)
    try:
        reader.run()
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return False
    finally:
        if hasattr(reader, "close"):
            reader.close()

    dye = params["dye_uptake_parameters"]
    actin = params["actin_parameters"]
    n = len(reader.tif_files)
    is_czi = config.paths.image_format == "czi"

    def channel(kind: str) -> str:
        """How a channel is selected, for messages."""
        if is_czi:
            number = {"membrane": dye["membrane_channel_number"], "dye": dye["dye_channel_number"],
                      "actin": actin["channel_number"]}[kind]
            names = getattr(reader, "channel_names", [])
            label = names[number - 1] if number and number <= len(names) and names[number - 1] else ""
            return f"channel {number}" + (f" '{label}'" if label else "")
        return "pattern '" + {"membrane": dye["membrane_channel_pattern"],
                              "dye": dye["dye_channel_pattern"],
                              "actin": actin["channel_pattern"]}[kind] + "'"

    logger.info("== SETUP CHECK ==")
    logger.info(f"Image format  : {config.paths.image_format}")
    logger.info(f"Data          : {getattr(reader, 'czi_path', None) or config.paths.data_folder}")
    logger.info(f"Results folder: {(output_root or default_output_root()) / config.paths.experiment_id}")
    logger.info(f"Membrane      : {n} frames ({channel('membrane')})")
    if dye["enable"]:
        logger.info(f"Dye           : {len(reader.dye_files)} frames ({channel('dye')})")
        if not reader.dye_files:
            problems.append(f"Dye analysis is enabled but no dye images were found ({channel('dye')}).")
    if actin["enable"]:
        logger.info(f"Actin         : {len(reader.actin_files)} frames ({channel('actin')})")
        if not reader.actin_files:
            problems.append(f"Actin analysis is enabled but no actin images were found ({channel('actin')}).")
        elif list(reader.actin_files) == list(reader.tif_files):
            notes.append("Actin is measured on the membrane images (same channel). Change the "
                         "actin channel setting if you have a separate actin channel.")

    pixel_um = getattr(reader, "pixel_size_um", None)
    if pixel_um:
        cfg_scale = params["experiment_parameters"]["scale_factor"]
        logger.info(f"Pixel size    : {pixel_um:.4f} µm (from the CZI file)")
        if abs(pixel_um - cfg_scale) > 0.02 * cfg_scale:
            notes.append(f"The CZI file says {pixel_um:.4f} µm per pixel but scale_factor is "
                         f"{cfg_scale}. The config value is used; check the objective.")

    if n < 2:
        problems.append("Fewer than 2 membrane frames found; nothing to analyse.")
    else:
        dt = (reader.time_data[-1] - reader.time_data[0]) / (n - 1)
        logger.info(f"Frame interval: {dt:.4f} s ({reader.metadata_method_used})")
        cfg_dt = params["experiment_parameters"]["frame_interval"]
        if reader.metadata_method_used and not reader.metadata_method_used.startswith("Manual"):
            if abs(dt - cfg_dt) > 0.1 * cfg_dt:
                notes.append(f"Image metadata says {dt:.4f} s per frame but the config's "
                             f"frame_interval is {cfg_dt} s. The metadata value is used.")

    first = utils.get_first_frame_number(reader.tif_files)
    logger.info(f"First frame number in file names: {first}")
    if dye["has_pulse"]:
        idx = utils.pulse_frame_to_index(dye["pulse_frame"], first)
        if 0 <= idx < n:
            logger.info(f"Pulse         : frame {dye['pulse_frame']:g} -> position {idx} "
                        f"(t = {reader.time_data[idx]:.2f} s)")
        else:
            problems.append(f"pulse_frame {dye['pulse_frame']:g} is outside the recording "
                            f"(only {n} frames).")
    else:
        logger.info("Pulse         : none (aspiration-only control)")

    name = experiment_naming.parse_experiment_name(config.paths.experiment_id)
    if name is None:
        notes.append(
            "experiment_id does not follow the naming rule "
            f"({experiment_naming.NAMING_RULE}), so the pulse settings come from the "
            "config instead of the name, and Bulk_Analysis will skip this experiment. "
            "See README.md.")
    else:
        logger.info(f"Experiment    : {name.describe()} (from the experiment name)")
        cfg_p = params["experiment_parameters"]["constant_pressure"]
        if abs(cfg_p - name.pressure_pa) > 1e-6:
            notes.append(f"constant_pressure is {cfg_p:g} Pa but the experiment name says "
                         f"{name.pressure_pa} Pa. The config value is used for the analysis.")

    cache = output_layout.first_existing(output_layout.session_cache_candidates(
        (output_root or default_output_root()) / config.paths.experiment_id))
    if cache.exists():
        import json
        try:
            cached_scale = json.loads(cache.read_text()).get('threshold_scale', 'legacy (CLAHE)')
        except (OSError, ValueError):
            cached_scale = None
        if cached_scale == threshold_scale(params):
            notes.append(f"A saved trap setup exists ({cache.name}); it will be reused. "
                         "Add --fresh to redo the interactive setup.")
        else:
            notes.append(f"A saved trap setup exists ({cache.name}) but its thresholds are on a "
                         f"different scale ({cached_scale or 'unreadable file'}; now: "
                         f"{threshold_scale(params)}), so the interactive setup will run again.")

    for note in notes:
        logger.warning(f"NOTE: {note}")
    for problem in problems:
        logger.error(f"PROBLEM: {problem}")
    if problems:
        logger.error(f"Setup check found {len(problems)} problem(s). Fix them before running.")
        return False
    logger.info("Setup check passed. Run again without --check to start the analysis.")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point. Returns a process exit code (0 = success)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    utils.setup_logging(debug_mode=args.debug)

    if args.init_config is not None:
        return 0 if init_config(args.init_config, args.copy_from,
                                args.data_folder, args.experiment_id) else 1

    if args.config_path is None:
        parser.print_help()
        return 2

    if args.batch is not None:
        from mfae.batch import run_batch   # imported here: mfae.batch uses this module
        return run_batch(args)

    try:
        listed = (config_loader.read_yaml(Path(args.config_path)).get("paths") or {}).get("data_folder")
    except config_loader.ConfigFileError:
        listed = None
    folder_list = isinstance(listed, list)
    if folder_list and args.data_folder is None:
        logger.error(f"{args.config_path} lists the folders for a batch: add --batch to analyse "
                     f"every experiment in them, or --data-folder \"<experiment>\" to analyse one.")
        return 1
    output_root = args.output_dir
    if folder_list:
        # One experiment with a batch config: same results folder and type
        # rules as the batch, so batch runs and the bulk analysis see it.
        from mfae.batch import setup_for_experiment
        setup = args.setup or setup_for_experiment(Path(args.data_folder), [Path(str(f)) for f in listed])
        output_root = Path(args.output_dir or default_output_root()) / setup
        logger.info(f"Results go to {output_root} (setup '{setup}')")

    config, layered = load_config(args.config_path, args.data_folder, args.experiment_id,
                                  type_from_name=folder_list)
    if config is None:
        return 1

    if args.show_config:
        print(show_config(config, layered))
        return 0

    if args.check:
        return 0 if check_setup(config, output_root) else 1

    # Find the image data before creating any output, so a wrong path or file
    # name stops the run with a short message instead of a traceback.
    if config.paths.image_format == "czi":
        from mfae.files.readers import resolve_czi_file
        try:
            import pylibCZIrw  # noqa: F401  (needed to read .czi files)
        except ImportError:
            logger.error("Reading .czi files needs the pylibCZIrw package, which isn't installed. "
                         "Install it (in Spyder: %pip install pylibCZIrw czifile, then restart "
                         "the kernel; in a terminal: pip install pylibCZIrw czifile).")
            return 1
        try:
            czi_file = resolve_czi_file(config.paths.data_folder, config.paths.experiment_id)
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        logger.info(f"CZI file: {czi_file}")

    if args.debug:
        config.workflow_settings.debug_mode = True

    if args.traps and args.fresh:
        logger.error("Use either --fresh (redo the whole setup) or --traps (redo single traps).")
        return 1

    analysis = None
    try:
        analysis = MFAAnalysis(config.paths.data_folder, str(config.paths.experiment_id), config,
                               output_root=output_root,
                               use_session_cache=not args.fresh,
                               **({'retune_traps': args.traps} if args.traps else {}))
        saved = save_effective_config(config, layered, args.config_path, analysis.results_dir)
        logger.info(f"Settings for this run saved to {saved}")
        return 0 if analysis.run_analysis() else 1
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user.")
        return 130
    except Exception:
        logger.error("Fatal error", exc_info=True)
        return 1
    finally:
        if analysis:
            analysis.cleanup()

