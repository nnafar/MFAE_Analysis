# -*- coding: utf-8 -*-
"""
Analyse one experiment (a .czi file or a folder of .tif images).

    python run_experiment.py config/my_exp.yaml            interactive setup, then analysis
    python run_experiment.py config/my_exp.yaml --check    check config and images (no windows)
    python run_experiment.py --init-config config/my_exp.yaml --data-folder "D:/data/<experiment>"
    python run_experiment.py --help                        all options

From Spyder:
    %runfile C:/GitHub/MFAE_Analysis/run_experiment.py --args config/my_exp.yaml --wdir

Results go to Output/<experiment>/ (README.txt there explains every file).
The code is in the mfae/ package; this script only starts it.
"""

import multiprocessing
import sys
from pathlib import Path

# Make the mfae package importable however this script is started
# (terminal, Spyder runfile, double-click), and in the worker processes.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mfae.cli import main  # noqa: E402

if __name__ == "__main__":
    multiprocessing.freeze_support()   # needed for parallel workers on Windows
    sys.exit(main())
