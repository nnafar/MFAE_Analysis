# -*- coding: utf-8 -*-
"""
Print how a Zeiss .czi file is organised, without copying any image data.

Used to prepare direct CZI support in the pipeline: paste the output into the
conversation (or an issue). It shows the dimensions, channel names and order,
pixel size, whether frame timestamps are stored, and checks that a frame can
be decoded.

Usage (on the computer that has the file):
    pip install pylibCZIrw czifile
    python tools/czi_info.py "D:/MFAE/<experiment>.czi"
"""

import argparse
import sys
from pathlib import Path


def _channels(meta):
    image = meta["ImageDocument"]["Metadata"]["Information"]["Image"]
    channels = image.get("Dimensions", {}).get("Channels", {}).get("Channel", [])
    if isinstance(channels, dict):
        channels = [channels]
    out = []
    for ch in channels:
        out.append({
            "name": ch.get("@Name"),
            "fluor": ch.get("Fluor"),
            "emission_nm": ch.get("EmissionWavelength"),
            "excitation_nm": ch.get("ExcitationWavelength"),
        })
    return image, out


def _scaling(meta):
    items = meta["ImageDocument"]["Metadata"].get("Scaling", {}).get("Items", {}).get("Distance", [])
    if isinstance(items, dict):
        items = [items]
    return {d.get("@Id"): float(d.get("Value")) * 1e6 for d in items if d.get("Value")}


def _timestamps(path):
    """Frame times from the TimeStamps attachment (czifile), or None."""
    try:
        import czifile
    except ImportError:
        return None, "czifile not installed (pip install czifile)"
    try:
        with czifile.CziFile(str(path)) as czi:
            for entry in czi.attachments():
                if entry.attachment_entry.name == "TimeStamps":
                    return list(entry.data()), None
        return None, "no TimeStamps attachment in the file"
    except Exception as e:  # report, don't fail: this is a diagnostic
        return None, f"could not read timestamps: {e}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("czi", type=Path, help="Path to the .czi file")
    args = parser.parse_args(argv)
    if not args.czi.exists():
        print(f"File not found: {args.czi}")
        return 1

    from pylibCZIrw import czi as pyczi

    print(f"File: {args.czi.name}  ({args.czi.stat().st_size / 1e9:.2f} GB)")
    with pyczi.open_czi(str(args.czi)) as czi:
        print(f"Dimensions (start, size): {czi.total_bounding_box}")
        try:
            print(f"Scenes: {czi.scenes_bounding_rectangle}")
        except Exception:
            pass
        meta = czi.metadata
        image, channels = _channels(meta)
        print(f"SizeT={image.get('SizeT')} SizeC={image.get('SizeC')} SizeZ={image.get('SizeZ')} "
              f"SizeS={image.get('SizeS')} SizeM={image.get('SizeM')} PixelType={image.get('PixelType')}")
        print("Channels (in file order, numbered from 1):")
        for i, ch in enumerate(channels, 1):
            print(f"  {i}: name={ch['name']!r} fluor={ch['fluor']!r} "
                  f"excitation={ch['excitation_nm']} emission={ch['emission_nm']}")
        print(f"Pixel size (µm): {_scaling(meta)}")
        acquired = meta["ImageDocument"]["Metadata"]["Information"].get("Image", {}).get("AcquisitionDateAndTime")
        print(f"Acquired: {acquired}")

        frame = czi.read(plane={"T": 0, "C": 0, "Z": 0})
        print(f"First frame decoded: shape={frame.shape} dtype={frame.dtype} "
              f"min={frame.min()} max={frame.max()}")

    times, problem = _timestamps(args.czi)
    if times is None:
        print(f"Frame timestamps: none ({problem})")
    else:
        t0 = times[0]
        rel = [t - t0 for t in times]
        steps = [b - a for a, b in zip(rel, rel[1:])]
        print(f"Frame timestamps: {len(times)} values; first 5 (s from start): "
              f"{[round(t, 3) for t in rel[:5]]}")
        if steps:
            print(f"  interval: mean {sum(steps) / len(steps):.4f} s, "
                  f"min {min(steps):.4f} s, max {max(steps):.4f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
