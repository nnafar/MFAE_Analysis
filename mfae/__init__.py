"""
MFAE analysis: microfluidic aspiration and electroporation of single cells.

Run it with the two scripts in the project folder:
    run_experiment.py      analyse one experiment (one .czi file or folder of .tif images)
    run_bulk_analysis.py   pool many analysed experiments into condition-level results

Package layout (one sub-package per step of the analysis):
    pipeline         runs one experiment: setup, per-trap workers, export
    cli              command line of run_experiment.py (--check, --init-config, ...)
    batch            --batch: every experiment under a setup folder, one config
    config/          settings: schema (validation), loader (defaults + your file), naming
    files/           reading images (.tif / .czi) and where results are written
    imaging/         trap boxes, rotation, interactive trap selection
    membrane/        protrusion and body masks, tip tracking, kymographs, tracking movies
    quantification/  dye uptake and actin, measured inside the membrane masks
    mechanics/       viscoelastic models, channel flow, fitting
    plotting/        per-experiment figures
    bulk/            pooling experiments: loading, mechanics, figures
    utils            shared image and plotting helpers
    paths            project, config and output folders
"""
