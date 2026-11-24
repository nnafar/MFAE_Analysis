# -*- coding: utf-8 -*-
"""
Layered configuration loading.

A run's settings are built from two YAML files, the second one winning:

    1. config/defaults.yaml            every setting, with the lab's tuned values
    2. the user's own config file      ~45 lines: paths, microscope, which
                                       analyses to run, plus any overrides

plus any command-line overrides. Finally, the pulse settings are read from the
experiment name (see experiment_naming.py): 0 V means no pulse, otherwise the
pulse frame is the name's `frame<N>`. A value set explicitly in the user's
file that contradicts the name is an error, so the two can't silently differ. The merged result is validated by
config_schema.MFAConfig exactly as a single file would be, so unknown keys and
bad values are still rejected. A full, old-style config file (every setting
listed) still works: it simply overrides all the defaults.

For traceability every leaf value remembers which layer set it, which is used
by --show-config, by validation messages and by the effective_config.yaml
saved with each run.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

import yaml

from experiment_naming import ExperimentName, NAMING_RULE, parse_experiment_name

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULTS_FILE = CONFIG_DIR / "defaults.yaml"


class ConfigFileError(Exception):
    """A config file is missing, unreadable, or not valid YAML. The message is user-facing."""


@dataclass
class LayeredConfig:
    data: Dict[str, Any]                        # merged settings (not yet validated)
    sources: Dict[str, str]                     # "section.key" -> layer label
    layers: List[str] = field(default_factory=list)  # labels, lowest priority first
    experiment: Optional[ExperimentName] = None       # parsed experiment name, if it follows the rule


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


def load_layers(user_path: Path, overrides: Optional[Dict[str, Any]] = None) -> LayeredConfig:
    """
    Merge defaults <- user file <- command-line overrides.
    Raises ConfigFileError for missing or invalid files.
    """
    user_path = Path(user_path)
    user = read_yaml(user_path)
    if "preset" in user:
        user.pop("preset")
        logger.warning(
            f"{user_path.name}: 'preset' is no longer used and is ignored. Whether there "
            f"was a pulse is read from the experiment name; put any other setting you "
            f"need directly in your config file. You can delete the 'preset:' line.")

    layers = [("defaults", read_yaml(DEFAULTS_FILE)), (user_path.name, user)]
    if overrides:
        layers.append(("command line", overrides))

    data: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    for label, layer in layers:
        data = deep_merge(data, layer)
        for key in _leaf_keys(layer):
            sources[key] = label

    layered = LayeredConfig(data=data, sources=sources,
                            layers=[label for label, _ in layers])
    _apply_experiment_name(layered, user_path,
                           explicit={user_path.name, "command line"})
    return layered


EXPERIMENT_NAME_LABEL = "experiment name"


def _apply_experiment_name(layered: LayeredConfig, user_path: Path, explicit: set) -> None:
    """Take the pulse settings from paths.experiment_id when it follows the naming rule."""
    experiment_id = (layered.data.get("paths") or {}).get("experiment_id")
    info = parse_experiment_name(experiment_id) if experiment_id else None
    layered.experiment = info
    if info is None:
        return

    derived = {"has_pulse": info.has_pulse}
    if info.has_pulse:
        derived["pulse_frame"] = info.pulse_frame

    dye = layered.data.setdefault("dye_uptake_parameters", {})
    for key, value in derived.items():
        dotted = f"dye_uptake_parameters.{key}"
        current = dye.get(key)
        if layered.sources.get(dotted) in explicit and current is not None:
            same = (float(current) == float(value)) if key == "pulse_frame" else (bool(current) == value)
            if not same:
                raise ConfigFileError(
                    f"{user_path}: dye_uptake_parameters.{key} is {current!r}, but the experiment "
                    f"name '{experiment_id}' says {value!r}. Remove the setting (it is read "
                    f"from the name) or fix the name ({NAMING_RULE}).")
            continue
        dye[key] = value
        layered.sources[dotted] = EXPERIMENT_NAME_LABEL


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
