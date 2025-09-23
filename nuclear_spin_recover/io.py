import numpy as np
import pandas as pd

from .spin_bath import SpinBath, NuclearSpin

def make_spinbath_from_Ivady_file(file_path, strong_thresh, weak_thresh, spin_type="C13"):
    """
    Initialize a SpinBath from an Ivady-style file with hyperfine couplings and positions.

    Parameters
    ----------
    file_path : str
        Path to text file with hyperfine couplings (MHz) and positions (Å).
    strong_thresh : float
        Threshold (kHz). Excludes all spins with any coupling component stronger than this.
    weak_thresh : float
        Threshold (kHz). Requires at least one coupling component stronger than this.
    spin_type : str, optional
        Spin species label (default: "C13").

    Returns
    -------
    SpinBath
        A SpinBath containing NuclearSpin objects constructed from file data.
    """
    # read file
    hf_data = pd.read_csv(file_path, sep=" ", header=None, names=[
        "distance",  # Å
        "x", "y", "z",  # Å
        "A_xx", "A_yy", "A_zz",  # MHz
        "A_xy", "A_xz", "A_yz"   # MHz
    ])

    # convert MHz → kHz
    MHz_to_kHz = 1000.0
    for col in ["A_xx", "A_yy", "A_zz", "A_xy", "A_xz", "A_yz"]:
        hf_data[col] *= MHz_to_kHz

    # compute A_perp and A_par
    hf_data["A_perp"] = np.sqrt(hf_data["A_xz"]**2 + hf_data["A_yz"]**2)
    hf_data["A_par"] = hf_data["A_zz"]

    # apply thresholds
    hf_df = hf_data[
        (hf_data["A_perp"] <= strong_thresh) &
        (np.abs(hf_data["A_par"]) <= strong_thresh)
    ]
    hf_df = hf_df[
        (hf_df["A_perp"] >= weak_thresh) |
        (np.abs(hf_df["A_par"]) >= weak_thresh)
    ]

    # build SpinBath
    spins = []
    for _, row in hf_df.iterrows():
        spin = NuclearSpin(
            spin_type,
            row["x"], row["y"], row["z"],
            A_xx=row["A_xx"], A_yy=row["A_yy"], A_zz=row["A_zz"],
            A_xy=row["A_xy"], A_yz=row["A_yz"], A_xz=row["A_xz"],
            A_par=row["A_par"], A_perp=row["A_perp"]
        )
        spins.append(spin)

    return SpinBath(spins)
