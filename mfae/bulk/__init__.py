"""
Bulk analysis: pools the per-experiment results into condition-level tables and figures.

    run.py          the bulk pipeline (run_bulk_analysis.py calls run.main)
    loader.py       finds and loads per-experiment results, groups cells by condition
    mechanics.py    per-cell fits (viscoelastic, model-independent, uptake)
    layout.py       folders and README of a bulk run
    figure_data.py  CSV export of every plotted value
    utils.py        shared style and helpers
    plots/          figure suites (aspiration mechanics, pulse, uptake, actin by fate)
"""
