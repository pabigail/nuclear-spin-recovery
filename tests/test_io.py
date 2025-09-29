import pytest
import numpy as np
import os
from pathlib import Path
import pickle
from nuclear_spin_recover import make_spinbath_from_Ivady_file


@pytest.fixture
def io_files_dir():
    """
    Return the real io_files directory path and Ivady file path,
    independent of the current working directory.
    """
    # Path to the directory containing this test file
    test_file_dir = Path(__file__).parent

    # io_files/ is assumed to be at the repo root (one level up from tests/)
    io_dir = test_file_dir.parent / "nuclear_spin_recover/io_files"
    nv_file = io_dir / "nv-2.txt"

    if not io_dir.exists():
        raise FileNotFoundError(f"io_files directory not found: {io_dir}")
    if not nv_file.exists():
        raise FileNotFoundError(f"Ivady file not found: {nv_file}")

    return io_dir, nv_file

@pytest.mark.parametrize("weak_thresh", [5, 10])
def test_existing_distance_matrix(io_files_dir, weak_thresh):
    """
    Test that make_spinbath_from_Ivady_file reads existing distance matrices
    and does not overwrite them.
    """
    io_dir, file_path = io_files_dir
    dist_file = io_dir / f"hf_dist_nv_low_{weak_thresh}_high_200.pkl"

    if not dist_file.exists():
        # Create dummy existing distance matrix
        with open(dist_file, "wb") as f:
            pickle.dump(np.eye(2), f)

    before_mtime = dist_file.stat().st_mtime

    bath = make_spinbath_from_Ivady_file(file_path, strong_thresh=200, weak_thresh=weak_thresh)

    after_mtime = dist_file.stat().st_mtime
    assert before_mtime == after_mtime, "Existing distance matrix should not be overwritten"

    # Check loaded matrix matches bath.distance_matrix
    with open(dist_file, "rb") as f:
        existing_matrix = pickle.load(f)
    assert np.allclose(existing_matrix, bath.distance_matrix)


def test_new_distance_matrix_written(io_files_dir):
    """
    Test that a new distance matrix file is written if it doesn't exist (weak_thresh=20)
    and remove it afterwards to keep repo clean.
    """
    io_dir, file_path = io_files_dir
    dist_file = io_dir / "hf_dist_nv_low_20_high_200.pkl"

    # Remove file if it exists to simulate "new" case
    file_existed_before = dist_file.exists()
    if file_existed_before:
        os.remove(dist_file)

    try:
        bath = make_spinbath_from_Ivady_file(file_path, strong_thresh=200, weak_thresh=20)

        # File should now exist
        assert dist_file.exists(), "Expected new distance matrix file to be written"

        # Distance matrix contents should match
        with open(dist_file, "rb") as f:
            loaded_matrix = pickle.load(f)
        assert np.allclose(loaded_matrix, bath.distance_matrix)

    finally:
        # Clean up: remove newly created distance matrix if it wasn't there before
        if not file_existed_before and dist_file.exists():
            os.remove(dist_file)
