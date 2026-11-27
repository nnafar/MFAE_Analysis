Files you don't edit:

  defaults.yaml        pipeline settings (image processing, detection, fitting, figures)
  template_tiff.yaml   starting point for a new config for exported .tif folders
  template_czi.yaml    starting point for a new config for Zeiss .czi files

Your configs live one folder up, in config/: config_asymmetric.yaml (.tif) and
config_symmetric.yaml (.czi). For a new setup, copy a template there with
  python run_experiment.py --init-config config/my_setup.yaml --data-folder "<folder or .czi file>"
See "How settings work" in README.md.
