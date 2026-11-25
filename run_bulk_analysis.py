# -*- coding: utf-8 -*-
"""
Pool many analysed experiments into condition-level tables and figures.

Run run_experiment.py on each experiment first, then:

    python run_bulk_analysis.py                          reads every experiment in Output/
    python run_bulk_analysis.py --output-root D:/MFAE/Output

From Spyder:
    %runfile C:/GitHub/MFAE_Analysis/run_bulk_analysis.py --wdir

Results go to Output/Bulk_Analysis_results/<YYMMDD_HHMM>/ (README.txt there
explains every file). The code is in mfae/bulk/; this script only starts it.
"""

import sys
from pathlib import Path

# Make the mfae package importable however this script is started.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mfae.bulk.run import main  # noqa: E402

if __name__ == "__main__":
    code = main()
    if hasattr(sys, "ps1") or "IPython" in sys.modules:
        # Interactive (Spyder / IPython): don't raise SystemExit, which shows
        # up there as "An exception has occurred"; just say how it ended.
        if code:
            print("\nFinished with problems (see the messages above).")
    else:
        sys.exit(code)
