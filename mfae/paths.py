# -*- coding: utf-8 -*-
"""Project folders, defined once: the project root, config/ and the default Output/ folder."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Output"
