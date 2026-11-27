# -*- coding: utf-8 -*-
"""
Reading Zeiss .czi files directly, on synthetic files written with
pylibCZIrw: channel 1 = dye (dim), channel 2 = membrane (a cell entering
the pipette), as in the lab's recordings.

Real Zen files can differ in ways synthetic ones can't reproduce (notably how
frame timestamps are stored); those paths fall back safely and are reported
by --check.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

pyczi = pytest.importorskip("pylibCZIrw.czi")

from mfae.files.readers import CziFrame, CziRead, resolve_czi_file, valid_timestamps  # noqa: E402
from mfae import utils  # noqa: E402

EXP_NAME = "260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-5ms-frame10"
N_FRAMES = 20
DYE_LEVEL = 30


def _membrane(t: int) -> np.ndarray:
    from synthetic_data_generators import create_synthetic_protrusion_image
    length = 0.5 if t < 4 else min(45.0, 8 + 1.5 * (t - 4))
    np.random.seed(t)          # same noise every time, so the frame can be rebuilt
    img, _ = create_synthetic_protrusion_image(protrusion_length_px=length)
    return img.astype(np.uint16) * 10


def write_czi(path: Path, n_frames: int = N_FRAMES, n_channels: int = 2) -> Path:
    with pyczi.create_czi(str(path), exist_ok=True) as czi:
        for t in range(n_frames):
            dye = np.full((50, 200), DYE_LEVEL + t, np.uint16)
            planes = [dye, _membrane(t)] + [dye] * (n_channels - 2)
            for c in range(n_channels):
                czi.write(data=planes[c], plane={"T": t, "C": c})
        czi.write_metadata(document_name=path.stem,
                           channel_names={0: "PI", 1: "CellMask"},
                           scale_x=0.315e-6, scale_y=0.315e-6)
    return path


@pytest.fixture
def czi_file(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    return write_czi(folder / f"{EXP_NAME}.czi")


CZI_PARAMS = {
    'paths': {'image_format': 'czi'},
    'experiment_parameters': {'frame_interval': 0.5},
    'dye_uptake_parameters': {'enable': True, 'membrane_channel_number': 2, 'dye_channel_number': 1},
    'actin_parameters': {'enable': True, 'channel_number': 2},
}


def test_reader_picks_channels_by_number(czi_file):
    reader = CziRead(czi_file, CZI_PARAMS)
    reader.run()
    try:
        assert len(reader.tif_files) == len(reader.dye_files) == N_FRAMES
        assert reader.channel_names == ["PI", "CellMask"]
        assert reader.pixel_size_um == pytest.approx(0.315)
        membrane = reader.read_img(reader.tif_files[10])
        dye = reader.read_img(reader.dye_files[10])
        assert membrane.shape == (50, 200) and membrane.ndim == 2
        assert np.array_equal(membrane, _membrane(10))          # channel 2 = membrane
        assert np.all(dye == DYE_LEVEL + 10)                     # channel 1 = dye
        # actin is in the membrane channel: same frames, reused
        assert reader.actin_files == reader.tif_files
    finally:
        reader.close()


def test_frame_numbers_count_from_one(czi_file):
    """`frame10` in the experiment name means the 10th frame, as for TIFFs from t0001."""
    reader = CziRead(czi_file, CZI_PARAMS)
    reader.run()
    reader.close()
    assert utils.get_first_frame_number(reader.tif_files) == 1
    assert utils.pulse_frame_to_index(10, utils.get_first_frame_number(reader.tif_files)) == 9
    assert reader.tif_files[9].name.endswith("[channel 2, frame 10]")


def test_without_timestamps_frame_interval_is_used(czi_file):
    reader = CziRead(czi_file, CZI_PARAMS)
    reader.run()
    reader.close()
    assert reader.metadata_method_used.startswith("Manual")
    assert reader.time_data[:3] == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("values, expected", [
    ([100.0, 100.5, 101.0], [0.0, 0.5, 1.0]),
    ([100.0, 100.5, 101.0, 101.5], [0.0, 0.5, 1.0]),   # extra values ignored
    ([100.0, 100.5], None),                            # too few
    ([100.0, 100.0, 101.0], None),                     # not increasing
    ([100.0, float("nan"), 101.0], None),
])
def test_timestamp_validation(values, expected):
    result = valid_timestamps(values, 3)
    assert result == (pytest.approx(expected) if expected else None)


def test_channel_number_must_exist(czi_file):
    params = {**CZI_PARAMS, 'dye_uptake_parameters': {**CZI_PARAMS['dye_uptake_parameters'],
                                                       'dye_channel_number': 3}}
    reader = CziRead(czi_file, params)
    with pytest.raises(ValueError, match="channels 1-2"):
        reader.run()
    reader.close()


def test_resolve_czi_file(tmp_path, czi_file):
    assert resolve_czi_file(czi_file) == czi_file
    assert resolve_czi_file(czi_file.parent) == czi_file          # folder with one .czi
    with pytest.raises(FileNotFoundError, match="No .czi"):
        resolve_czi_file(tmp_path)
    write_czi(czi_file.parent / "other.czi", n_frames=2)
    with pytest.raises(FileNotFoundError, match="2 .czi files"):
        resolve_czi_file(czi_file.parent)


def test_frames_cached_for_workers_match_the_file(tmp_path, czi_file):
    """The shared-memory stack the workers read is built correctly from a CZI."""
    from mfae import cli, pipeline
    from mfae.config import loader as config_loader
    from mfae.config.schema import MFAConfig
    cfg = MFAConfig(**config_loader.load_layers(
        ROOT / "config" / "pipeline/template_czi.yaml",
        {"paths": {"data_folder": str(czi_file), "experiment_id": EXP_NAME}}).data)
    analysis = pipeline.MFAAnalysis(czi_file, EXP_NAME, cfg, output_root=tmp_path / "out")
    try:
        analysis.file_reader.run()
        path, shape, dtype = analysis._prepare_shared_memory_stack(
            analysis.file_reader.tif_files, "Membrane")
        stack = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        assert shape == (N_FRAMES, 50, 200)
        assert np.array_equal(stack[7], _membrane(7))
        del stack
    finally:
        analysis.cleanup()


# -----------------------------------------------------------------------------
# Configs and command line
# -----------------------------------------------------------------------------

def test_init_and_check_a_czi_experiment(tmp_path, czi_file, capsys):
    from mfae import cli, pipeline
    dest = tmp_path / "exp.yaml"
    assert cli.main(["--init-config", str(dest), "--data-folder", str(czi_file)]) == 0
    data = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert data["paths"]["image_format"] == "czi"                 # czi template chosen
    assert data["paths"]["experiment_id"] == EXP_NAME              # from the file name
    capsys.readouterr()
    assert cli.main([str(dest), "--check"]) == 0
    out = capsys.readouterr().out
    assert "Membrane      : 20 frames (channel 2 'CellMask')" in out
    assert "Dye           : 20 frames (channel 1 'PI')" in out
    assert "Pulse         : frame 10 -> position 9" in out
    assert "Pixel size    : 0.3150" in out


def test_tiff_template_refused_for_a_czi_file(tmp_path, czi_file):
    from mfae import cli, pipeline
    dest = tmp_path / "exp.yaml"
    assert cli.main(["--init-config", str(dest), "--data-folder", str(czi_file),
                     "--from", str(ROOT / "config" / "pipeline/template_tiff.yaml")]) == 1
    assert not dest.exists()


def test_format_mismatch_is_reported(tmp_path, czi_file):
    from mfae import cli, pipeline
    # TIFF config pointed at a .czi file
    config, _ = cli.load_config(ROOT / "config" / "pipeline/template_tiff.yaml", data_folder=czi_file)
    assert config is None


def test_channel_settings_of_the_other_format_are_refused(tmp_path, czi_file):
    from mfae.config import loader as config_loader
    text = (ROOT / "config" / "pipeline/template_czi.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("  dye_channel_number: 1\n", "  dye_channel_pattern: 'C2'\n"),
                   encoding="utf-8")
    with pytest.raises(config_loader.ConfigFileError, match="only used with image_format: tiff"):
        config_loader.load_layers(bad, {"paths": {"data_folder": str(czi_file)}})


def test_folder_of_many_czi_files_uses_the_experiment_name(tmp_path, czi_file):
    """One folder can hold many experiments; the file named after experiment_id is used."""
    other = "261119_MDAMB231_LifeAct_Chip1_Experiment5-1100Pa-50V-1ms-frame300"
    write_czi(czi_file.parent / f"{other}.czi", n_frames=3)
    assert resolve_czi_file(czi_file.parent, EXP_NAME) == czi_file
    assert resolve_czi_file(czi_file.parent, other).stem == other
    with pytest.raises(FileNotFoundError, match="none is named after the experiment"):
        resolve_czi_file(czi_file.parent, "260101_Unknown_X_Y_Z-1Pa-0V-0ms-frame0")


def test_missing_czi_stops_before_any_output(tmp_path, czi_file, capsys):
    from mfae import cli, pipeline
    write_czi(czi_file.parent / "another.czi", n_frames=2)
    dest = tmp_path / "exp.yaml"
    cli.main(["--init-config", str(dest), "--data-folder", str(czi_file)])
    text = dest.read_text(encoding="utf-8")
    dest.write_text(text.replace(str(czi_file), str(czi_file.parent)).replace(
        EXP_NAME, "260101_MDAMB231_WT_Chip1_Experiment9-1100Pa-100V-1ms-frame10"), encoding="utf-8")
    out_dir = tmp_path / "out"
    capsys.readouterr()
    assert cli.main([str(dest), "--output-dir", str(out_dir)]) == 1
    assert "none is named after the experiment" in capsys.readouterr().out
    assert not out_dir.exists()
