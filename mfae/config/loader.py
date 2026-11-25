# -*- coding: utf-8 -*-
"""
Configuration loading: two files that never share a setting.

    config/defaults.yaml       pipeline settings, for developers only
    the user's config          the experiment: experiment_type, paths,
                               microscope, which analyses to run, channel
                               names, interactive setup, CPU cores
                               (exactly user_keys_for(image_format))

A user config must set every key in USER_KEYS plus the channel settings for
its image format (CHANNEL_KEYS: name patterns for TIFF, channel numbers for
CZI); OPTIONAL_USER_KEYS may be set; nothing else is allowed. defaults.yaml
must contain none of these. Because they
never overlap, either file can be rewritten without affecting the other.
Command-line overrides (data folder, experiment name) are user settings too.

The pulse: `experiment_type` (ep / asp) says whether there was one; its
voltage, duration and frame are read from the experiment name (see
mfae/config/naming.py). A type that contradicts the name's voltage is an
error.

Full configs - files that list every setting, such as old 400-line configs
and the effective_config.yaml saved with each run - are recognised by having
no `experiment_type` and are loaded on their own, as before.

Every leaf value remembers which file set it; this is used by --show-config,
validation messages and effective_config.yaml. The result is validated by
config_schema.MFAConfig.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

import yaml

from mfae.config.naming import ExperimentName, NAMING_RULE, parse_experiment_name

logger = logging.getLogger(__name__)

from mfae.paths import CONFIG_DIR  # noqa: E402  (project config/ folder)
DEFAULTS_FILE = CONFIG_DIR / "defaults.yaml"

EXPERIMENT_TYPES = ("ep", "asp")   # electroporation, aspiration only

# Everything a user config contains, and nothing else. defaults.yaml must not
# contain any of these.
USER_KEYS = (
    "experiment_type",
    # Experiment
    "paths.image_format",
    "paths.data_folder",
    "paths.experiment_id",
    "experiment_parameters.scale_factor",
    "experiment_parameters.constant_pressure",
    "experiment_parameters.frame_interval",
    "experiment_parameters.max_traps",
    # Analyses (on/off) and channels
    "dye_uptake_parameters.enable",
    "actin_parameters.enable",
    "rupture_detection.enable",
    "rupture_detection.enable_smoothing",
    "rupture_detection.enable_outlier_rejection",
    "model_parameters.perform_fitting",
    "workflow_settings.create_kymographs",
    # Interactive setup
    "workflow_settings.selection_frame_fraction",
    "workflow_settings.default_roi_width_um",
    "workflow_settings.default_roi_height_um",
    # Computer
    "workflow_settings.n_workers",
)

IMAGE_FORMATS = ("tiff", "czi")

# How the channels are chosen depends on the image format; a user config
# contains the set for its format (required) and not the other one.
CHANNEL_KEYS = {
    # TIFF: text in the file names (e.g. 'C1').
    "tiff": ("dye_uptake_parameters.membrane_channel_pattern",
             "dye_uptake_parameters.dye_channel_pattern",
             "actin_parameters.channel_pattern"),
    # CZI: channel numbers in the file, counted from 1 as Fiji shows them.
    "czi": ("dye_uptake_parameters.membrane_channel_number",
            "dye_uptake_parameters.dye_channel_number",
            "actin_parameters.channel_number"),
}


def user_keys_for(image_format: str) -> tuple:
    """Every setting a user config with this image format must contain."""
    return USER_KEYS + CHANNEL_KEYS[image_format]


# User settings that may be given but are not required.
OPTIONAL_USER_KEYS = (
    # Only for experiment names that don't follow the naming rule.
    "dye_uptake_parameters.pulse_frame",
)

# Kept as an alias for code that imported the old name.
EXPERIMENT_KEYS = USER_KEYS


class ConfigFileError(Exception):
    """A config file is missing, invalid, or breaks the two-file rules. The message is user-facing."""


@dataclass
class LayeredConfig:
    data: Dict[str, Any]                        # merged settings (not yet validated)
    sources: Dict[str, str]                     # "section.key" -> file label
    layers: List[str] = field(default_factory=list)  # labels, lowest priority first
    experiment: Optional[ExperimentName] = None       # parsed experiment name, if it follows the rule
    experiment_type: Optional[str] = None             # 'ep' / 'asp'; None for a full config


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML mapping, turning every failure into a readable ConfigFileError."""
    path = Path(path)
    if not path.exists():
        raise ConfigFileError(f"Config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigFileError(
            f"{path} is not valid YAML (check indentation and quotes):\n{e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigFileError(f"{path} must contain sections like 'paths:', not a single value.")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict: nested dicts are merged key by key, anything else is replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _leaf_keys(data: Dict[str, Any], prefix: str = "") -> List[str]:
    keys = []
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            keys.extend(_leaf_keys(value, dotted + "."))
        else:
            keys.append(dotted)
    return keys


def _bullets(keys: List[str]) -> str:
    return "\n".join(f"  - {k}" for k in keys)


def all_user_keys() -> set:
    """Every setting that belongs to users, for any image format."""
    keys = set(USER_KEYS) | set(OPTIONAL_USER_KEYS)
    for channel_keys in CHANNEL_KEYS.values():
        keys |= set(channel_keys)
    return keys


def read_defaults() -> Dict[str, Any]:
    """The developer settings; refuses a defaults file that contains user settings."""
    defaults = read_yaml(DEFAULTS_FILE)
    user_owned = all_user_keys()
    misplaced = [k for k in _leaf_keys(defaults) if k in user_owned]
    if misplaced:
        raise ConfigFileError(
            f"{DEFAULTS_FILE} contains user settings, which belong in the experiment "
            f"configs instead:\n{_bullets(misplaced)}")
    return defaults


def load_layers(user_path: Path, overrides: Optional[Dict[str, Any]] = None,
                experiment_type_override: Optional[str] = None) -> LayeredConfig:
    """
    Combine config/defaults.yaml with a user config (or load a full config on
    its own), apply command-line overrides and the experiment name.
    experiment_type_override ('ep' / 'asp') replaces the config's
    experiment_type: batch runs use it so each experiment's type follows its
    name (0 V = asp) while a whole setup shares one config.
    Raises ConfigFileError with a user-facing message on any problem.
    """
    user_path = Path(user_path)
    if user_path.exists() and user_path.resolve() == DEFAULTS_FILE.resolve():
        raise ConfigFileError(
            f"{user_path} holds the pipeline (developer) settings and can't be run on its "
            f"own. Give an experiment config instead, e.g. config/config_tiff_template.yaml.")
    user = read_yaml(user_path)
    if "preset" in user:
        user.pop("preset")
        logger.warning(
            f"{user_path.name}: 'preset' is no longer used and is ignored; use "
            f"'experiment_type: ep' or 'experiment_type: asp'. You can delete the 'preset:' line.")
    defaults = read_defaults()

    override_entries = user.pop("overrides", None) or []
    experiment_type = user.pop("experiment_type", None)
    if experiment_type is not None and experiment_type_override is not None:
        experiment_type = experiment_type_override
    if experiment_type is None:
        # Full config (old-style, or a saved effective_config.yaml): stands on its own.
        if override_entries:
            raise ConfigFileError(f"{user_path}: 'overrides' only works in an experiment or setup "
                                  f"config (one with experiment_type), not in a full config.")
        layers = [(user_path.name, user)]
    else:
        experiment_type = str(experiment_type).lower()
        if experiment_type not in EXPERIMENT_TYPES:
            raise ConfigFileError(
                f"{user_path}: experiment_type must be 'ep' (electroporation) or 'asp' "
                f"(aspiration only), not '{experiment_type}'.")
        image_format = str((user.get("paths") or {}).get("image_format", "")).lower()
        if image_format not in IMAGE_FORMATS:
            raise ConfigFileError(
                f"{user_path}: paths.image_format must be 'tiff' (folder of .tif files) or "
                f"'czi' (Zeiss .czi file), not '{image_format or '(missing)'}'.")
        required = user_keys_for(image_format)
        allowed = set(required) | set(OPTIONAL_USER_KEYS)
        given = set(_leaf_keys(user)) | {"experiment_type"}
        other = [f for f in IMAGE_FORMATS if f != image_format][0]
        wrong_format = sorted(given & set(CHANNEL_KEYS[other]))
        if wrong_format:
            raise ConfigFileError(
                f"{user_path} has image_format: {image_format}, but these channel settings are "
                f"only used with image_format: {other}:\n{_bullets(wrong_format)}\n"
                f"Use instead:\n{_bullets(list(CHANNEL_KEYS[image_format]))}")
        developer = sorted(given - allowed)
        if developer:
            raise ConfigFileError(
                f"{user_path} contains setting(s) that are not user settings:\n"
                f"{_bullets(developer)}\n"
                f"Pipeline settings live only in {DEFAULTS_FILE.relative_to(CONFIG_DIR.parent)} "
                f"(developers). If this is a typo, compare with config/config_tiff_template.yaml.")
        override_keys = set(_leaf_keys(overrides or {}))
        missing = [k for k in required if k not in given and k not in override_keys]
        if missing:
            raise ConfigFileError(
                f"{user_path} is missing user setting(s) that every config must set:\n"
                f"{_bullets(missing)}\nCopy them from config/config_tiff_template.yaml (or config_czi_template.yaml) "
                f"(or start from it with --init-config).")
        layers = [("defaults", defaults), (user_path.name, user)]
        layers += override_layers(user_path, override_entries, user, overrides, allowed)

    if overrides:
        layers.append(("command line", overrides))

    data: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    for label, layer in layers:
        data = deep_merge(data, layer)
        for key in _leaf_keys(layer):
            sources[key] = label

    layered = LayeredConfig(data=data, sources=sources, experiment_type=experiment_type,
                            layers=[label for label, _ in layers])
    _apply_pulse(layered, user_path, explicit={user_path.name, "command line"})
    return layered


def override_layers(user_path: Path, entries: Any, user: Dict[str, Any],
                    cli_overrides: Optional[Dict[str, Any]], allowed: set) -> List[Tuple[str, Dict[str, Any]]]:
    """
    The 'overrides' entries of a config that apply to this experiment.

    A setup config can hold settings that differ for some of its experiments:

        overrides:
          - match: 100us_100V              # condition folder ...
            rupture_detection:
              enable_smoothing: True
          - match: [Experiment272, Experiment283]   # ... or part of experiment names
            experiment_parameters:
              frame_interval: 1.72

    An entry applies when one of its `match` texts appears in the experiment's
    name or anywhere in its data path (case-insensitive). Entries are applied
    in order on top of the config, so a later entry wins. They may change any
    user setting except the paths (a batch sets those per experiment).
    """
    if not entries:
        return []
    if not isinstance(entries, list):
        raise ConfigFileError(f"{user_path}: 'overrides' must be a list of entries, each with "
                              f"'match:' and the settings to change (see config_asymmetric_tiff.yaml).")
    paths = {**(user.get("paths") or {}), **((cli_overrides or {}).get("paths") or {})}
    targets = [str(paths.get("experiment_id", "")), str(paths.get("data_folder", "")).replace("\\", "/")]
    targets = [t.lower() for t in targets if t]
    out = []
    for n, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or "match" not in entry:
            raise ConfigFileError(f"{user_path}: overrides entry {n} needs a 'match:' line "
                                  f"(part of the experiment name or folder it applies to).")
        entry = dict(entry)
        match = entry.pop("match")
        patterns = [str(m) for m in (match if isinstance(match, list) else [match]) if str(m).strip()]
        if not patterns:
            raise ConfigFileError(f"{user_path}: overrides entry {n} has an empty 'match:'.")
        keys = set(_leaf_keys(entry))
        bad = sorted(k for k in keys if k not in allowed or k.startswith("paths."))
        if bad:
            raise ConfigFileError(
                f"{user_path}: overrides entry {n} (match: {', '.join(patterns)}) changes setting(s) "
                f"that can't be overridden:\n{_bullets(bad)}\nOverrides may change user settings "
                f"except paths (compare with the settings above them in the file).")
        if not keys:
            continue
        if any(p.lower() in t for p in patterns for t in targets):
            out.append((f"{user_path.name} overrides[{', '.join(patterns)}]", entry))
    return out


EXPERIMENT_NAME_LABEL = "experiment name"
EXPERIMENT_TYPE_LABEL = "experiment_type"


def _apply_pulse(layered: LayeredConfig, user_path: Path, explicit: set) -> None:
    """
    Set dye_uptake_parameters.has_pulse / pulse_frame from experiment_type and
    the experiment name, refusing combinations that contradict each other.
    """
    experiment_id = (layered.data.get("paths") or {}).get("experiment_id")
    info = parse_experiment_name(experiment_id) if experiment_id else None
    layered.experiment = info
    dye = layered.data.setdefault("dye_uptake_parameters", {})

    def set_value(key: str, value: Any, label: str) -> None:
        dye[key] = value
        layered.sources[f"dye_uptake_parameters.{key}"] = label

    def explicit_value(key: str) -> Any:
        if layered.sources.get(f"dye_uptake_parameters.{key}") in explicit:
            return dye.get(key)
        return None

    # --- User config: experiment_type decides whether there was a pulse ---
    if layered.experiment_type is not None:
        has_pulse = layered.experiment_type == "ep"
        set_value("has_pulse", has_pulse, EXPERIMENT_TYPE_LABEL)
        if info is not None and info.has_pulse != has_pulse:
            raise ConfigFileError(
                f"{user_path}: experiment_type is '{layered.experiment_type}' but the experiment "
                f"name says {info.voltage_v} V ({'a pulse' if info.has_pulse else 'no pulse'}). "
                f"Use experiment_type: {'ep' if info.has_pulse else 'asp'}, or fix the name.")
        if not has_pulse:
            return
        given = explicit_value("pulse_frame")
        if info is not None:
            if given is not None and float(given) != float(info.pulse_frame):
                raise ConfigFileError(
                    f"{user_path}: pulse_frame is {given!r} but the experiment name says "
                    f"frame{info.pulse_frame}. Remove pulse_frame (it is read from the name) "
                    f"or fix the name.")
            set_value("pulse_frame", info.pulse_frame, EXPERIMENT_NAME_LABEL)
        elif given is None:
            raise ConfigFileError(
                f"{user_path}: the experiment name '{experiment_id}' doesn't follow the naming "
                f"rule ({NAMING_RULE}), so the pulse frame can't be read from it. Rename it, "
                f"or add dye_uptake_parameters.pulse_frame to the config.")
        return

    # --- Full config: it states the pulse itself; the name must agree ---
    if info is None:
        needed = ["has_pulse"] + (["pulse_frame"] if dye.get("has_pulse") else [])
        missing = [f"dye_uptake_parameters.{k}" for k in needed if explicit_value(k) is None]
        if missing:
            raise ConfigFileError(
                f"{user_path}: the experiment name '{experiment_id}' doesn't follow the naming "
                f"rule ({NAMING_RULE}), so the pulse can't be read from it. Rename it, or set "
                f"{' and '.join(missing)} in the config.")
        return
    derived = {"has_pulse": info.has_pulse}
    if info.has_pulse:
        derived["pulse_frame"] = info.pulse_frame
    for key, value in derived.items():
        current = explicit_value(key)
        if current is not None:
            same = float(current) == float(value) if key == "pulse_frame" else bool(current) == value
            if not same:
                raise ConfigFileError(
                    f"{user_path}: dye_uptake_parameters.{key} is {current!r}, but the experiment "
                    f"name '{experiment_id}' says {value!r}. Fix one of them ({NAMING_RULE}).")
            continue
        set_value(key, value, EXPERIMENT_NAME_LABEL)


# =============================================================================
# OUTPUT
# =============================================================================

def _yaml_scalar(value: Any) -> str:
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=1000).strip()
    return text[:-4].strip() if text.endswith("...") else text


def render_settings(settings: Dict[str, Any], sources: Optional[Dict[str, str]] = None,
                    hide_source: str = "") -> str:
    """
    Settings as YAML text with a trailing `# <layer>` comment on every value
    that did not come from `hide_source`. Used by --show-config and for
    effective_config.yaml.
    """
    sources = sources or {}
    lines: List[str] = []

    def emit(data: Dict[str, Any], indent: int, prefix: str) -> None:
        for key, value in data.items():
            dotted = f"{prefix}{key}"
            pad = "  " * indent
            if isinstance(value, dict) and value:
                if indent == 0 and lines:
                    lines.append("")
                lines.append(f"{pad}{key}:")
                emit(value, indent + 1, dotted + ".")
            else:
                source = sources.get(dotted, "")
                note = f"  # {source}" if source and source != hide_source else ""
                lines.append(f"{pad}{key}: {_yaml_scalar(value)}{note}")

    emit(settings, 0, "")
    return "\n".join(lines) + "\n"
