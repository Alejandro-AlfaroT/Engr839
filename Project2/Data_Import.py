"""Import acceleration time histories from 839WalkingExperiments.mat.

Dependencies: numpy, scipy, matplotlib
Install if needed: python -m pip install numpy scipy matplotlib

Run this script in your Python IDE, or with:
    python import_walking_data.py

Edit the settings below to choose a channel or plot interval. The full selected
channel is available as `time_seconds` and `acceleration` after running in an IDE.
Only one channel is loaded, because the complete MAT file is about 4.4 GB.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, whosmat


# -------------------------- EDIT THESE SETTINGS ---------------------------
MAT_FILE = Path(r"C:\Users\andro\Downloads\839WalkingExperiments.mat")
CHANNEL = 1                         # Choose a channel from 1 to 20.
CONVERT_TO_MS2 = False               # False: g; True: m/s^2.
PLOT_START_SECONDS = 0.0
PLOT_END_SECONDS = 660.0              # Preview interval; full data stays loaded.
# -------------------------------------------------------------------------


def load_channel(mat_file, channel=1, convert_to_ms2=False):
    r"""Return (time_seconds, acceleration, metadata) for one complete channel.

    Example from another script or notebook in this folder:
        from import_walking_data import load_channel
        t, a, info = load_channel(r"C:\path\839WalkingExperiments.mat", channel=3)

    The source file stores acceleration in g. Its sensitivity metadata is not
    applied again. No filtering, detrending, resampling, or time shifts are made.
    """
    mat_file = Path(mat_file)
    if not mat_file.is_file():
        raise FileNotFoundError(f"MAT file not found: {mat_file}")

    # Read names and shapes without loading the large signal matrices.
    matches = [
        name for name, shape, dtype in whosmat(mat_file)
        if re.fullmatch(rf"Ch{channel}_StreamSegment\d+_\d+", name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one stream for channel {channel}; found {len(matches)}. "
            "This file has channels 1 through 20."
        )
    variable_name = matches[0]
    suffix = f"_Ch{channel}"
    metadata_names = [
        "SamplingRate", "XQuantityOf" + suffix, "XUnityOf" + suffix,
        "YQuantityOf" + suffix, "YUnityOf" + suffix,
    ]
    data = loadmat(
        mat_file, variable_names=[variable_name, *metadata_names],
        squeeze_me=True,
    )
    values = data[variable_name]
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected [time, acceleration] columns; got {values.shape}.")
    if (data["XQuantityOf" + suffix] != "Time"
            or data["XUnityOf" + suffix] != "s"
            or data["YQuantityOf" + suffix] != "Acceleration"):
        raise ValueError("The selected channel is not acceleration versus seconds.")

    # MATLAB column 1 -> Python index 0; MATLAB column 2 -> Python index 1.
    time_seconds = values[:, 0]
    acceleration = values[:, 1]
    acceleration_unit = str(data["YUnityOf" + suffix])
    if convert_to_ms2:
        if acceleration_unit != "g":
            raise ValueError(f"Expected acceleration in g; got {acceleration_unit!r}.")
        acceleration = acceleration * 9.80665
        acceleration_unit = "m/s^2"

    metadata = {
        "channel": channel,
        "variable_name": variable_name,
        "sampling_rate_hz": float(data["SamplingRate"]),
        "acceleration_unit": acceleration_unit,
        "sample_count": time_seconds.size,
        "start_seconds": float(time_seconds[0]),
        "end_seconds": float(time_seconds[-1]),
    }
    return time_seconds, acceleration, metadata


if __name__ == "__main__":
    time_seconds, acceleration, metadata = load_channel(
        MAT_FILE, CHANNEL, convert_to_ms2=CONVERT_TO_MS2,
    )
    print(f"Loaded {metadata['variable_name']}")
    print(f"Samples: {metadata['sample_count']:,}")
    print(f"Sampling rate: {metadata['sampling_rate_hz']:g} Hz")
    print(f"Time: {metadata['start_seconds']:.6f} to {metadata['end_seconds']:.6f} s")
    print(f"Acceleration unit: {metadata['acceleration_unit']}")
    print("Full arrays: time_seconds, acceleration")

    # Slice only the plot. The imported arrays above retain every sample.
    if PLOT_END_SECONDS <= PLOT_START_SECONDS:
        raise ValueError("PLOT_END_SECONDS must be greater than PLOT_START_SECONDS.")
    first = np.searchsorted(time_seconds, PLOT_START_SECONDS, side="left")
    last = np.searchsorted(time_seconds, PLOT_END_SECONDS, side="right")
    if last - first < 2:
        raise ValueError("The selected plot interval contains fewer than two samples.")

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(time_seconds[first:last], acceleration[first:last], linewidth=0.7)
    ax.set(
        xlabel="Time (s)",
        ylabel=f"Acceleration ({metadata['acceleration_unit']})",
        title=f"Walking experiment - Channel {CHANNEL}",
        xlim=(PLOT_START_SECONDS, PLOT_END_SECONDS),
    )
    ax.grid(True, alpha=0.3)
    plt.show()
