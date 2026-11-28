# -*- coding: utf-8 -*-
"""
Export the trap images of an analysed experiment, losslessly, to one small file.

TEMPORARY helper for developing the detection on a dataset whose raw file is
too large to share. Delete this file when it is no longer needed; nothing in
the pipeline uses it.

It writes exactly what the analysis works on: every frame rotated and cropped
to each trap box as set up in the interactive setup, at the original bit
depth (16-bit, no compression loss), plus the saved setup of each trap
(entrance, thresholds, smoothing), the frame times and the pixel size. One
trap's membrane channel is about 4 MB.

Usage (on the computer with the data, after the experiment has been analysed):

    python tools/export_trap_stacks.py "Output/<setup>/<experiment>" --traps 1,5,6

Spyder:

    %runfile C:/GitHub/MFAE_Analysis/tools/export_trap_stacks.py --args "Output/<setup>/<experiment> --traps 1,5,6" --wdir

Options: --channels membrane dye (default: membrane only), --data-folder PATH
if the raw data has moved since the analysis, --out FILE (default:
<experiment>_traps_<list>.npz in the results folder).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export trap images (lossless) of an analysed experiment.")
    parser.add_argument("results", type=Path, help="The experiment's results folder (Output/<setup>/<experiment>).")
    parser.add_argument("--traps", required=True,
                        help="Trap numbers as in the movies and plots, e.g. 1,5,6.")
    parser.add_argument("--channels", nargs="+", default=["membrane"], choices=["membrane", "dye"])
    parser.add_argument("--data-folder", type=Path, default=None,
                        help="The raw data (.czi file or .tif folder), if it moved since the analysis.")
    parser.add_argument("--out", type=Path, default=None, help="Output file (.npz).")
    args = parser.parse_args(argv)

    from mfae import utils
    from mfae.cli import load_config
    from mfae.files import output_layout
    from mfae.files.readers import make_file_reader

    results = args.results.resolve()
    layout = output_layout.build(results)
    cfg_path = output_layout.experiment_file(layout, 'effective_config')
    cache_path = cfg_path.parent / "session_cache.json"
    if not cache_path.exists():
        cache_path = results / "session_cache.json"            # older results
    for p in (cfg_path, cache_path):
        if not p.exists():
            print(f"Not found: {p}. Give the results folder of an analysed experiment.")
            return 1

    config, _ = load_config(cfg_path, args.data_folder)
    if config is None:
        return 1
    params = config.model_dump()
    data = Path(args.data_folder or config.paths.data_folder)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    angle = float(cache.get("rotation_angle", 0.0))
    by_trap = {int(c["trap_index"]) + 1: c for c in cache.get("trap_configs", [])}
    traps = sorted({int(t) for t in str(args.traps).replace(" ", "").split(",") if t})
    missing = [t for t in traps if t not in by_trap]
    if missing:
        print(f"Trap(s) {missing} are not in the saved setup; available: {sorted(by_trap)}")
        return 1

    reader = make_file_reader(data, params)
    reader.run()                               # finds the frames of each channel and their times
    lists = {"membrane": list(reader.tif_files), "dye": list(getattr(reader, "dye_files", []) or [])}
    out = {}
    for ch in args.channels:
        frames = lists[ch]
        if not frames:
            print(f"No {ch} frames found in {data}; skipped.")
            continue
        stacks = {t: [] for t in traps}
        prev = None
        for k, f in enumerate(frames):
            img = reader.read_img(f)
            if img is None:                        # unreadable frame: zeros, as the pipeline does
                if prev is None:
                    print(f"Could not read the first {ch} frame ({f}).")
                    return 1
                img = np.zeros_like(prev)
            if img.ndim == 3:
                img = img[..., 0]
            prev = img
            rot = utils.rotate_image(img, angle) if angle else img
            for t in traps:
                y0, y1, x0, x1 = by_trap[t]["roi_coords"]
                stacks[t].append(np.array(rot[y0:y1, x0:x1]))
            if k % 50 == 0:
                print(f"  {ch}: frame {k + 1}/{len(frames)}")
        for t in traps:
            out[f"trap{t:02d}_{ch}"] = np.stack(stacks[t])

    if not any(k.startswith("trap") for k in out):
        print("Nothing exported.")
        return 1
    times = getattr(reader, "time_data", None)
    meta = {
        "experiment_id": str(config.paths.experiment_id),
        "scale_factor_um_per_px": float(params.get("experiment_parameters", {}).get("scale_factor", 0.0)),
        "rotation_angle": angle,
        "threshold_scale": cache.get("threshold_scale"),
        "traps": {str(t): by_trap[t] for t in traps},
        "image_processing": params.get("image_processing", {}),
    }
    out["time_s"] = np.asarray(times if times is not None else [], dtype=float)
    out["meta_json"] = np.array(json.dumps(meta, default=str))

    name = f"{results.name}_traps_{'-'.join(str(t) for t in traps)}.npz"
    dest = args.out or (results / name)
    np.savez_compressed(dest, **out)
    size = dest.stat().st_size / 1e6
    print(f"Saved {dest} ({size:.1f} MB): " + ", ".join(f"{k} {v.shape}" for k, v in out.items()
                                                         if k.startswith("trap")))
    print("Share this file; it can be deleted afterwards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
