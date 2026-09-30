"""Plot one-sided FFT amplitude spectra for the walking experiment channels.

Run this file in PyCharm, or with: python FFT_Analysis.py
Requires numpy, scipy, matplotlib, and Data_Import.py in the same folder.
Plots are saved in outputs/frequency_domain next to this script.

The FFT uses only the selected time interval. By default, the signal mean is
removed and a periodic Hann window is applied. Amplitudes are corrected for
the window's coherent gain; these are peak amplitude spectra, not PSDs or RMS
spectra. Off-bin tones can still have spectral leakage and amplitude error.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann

from Data_Import import load_channel


# -------------------------- EDIT THESE SETTINGS ---------------------------
MAT_FILE = Path(r"D:\839WalkingExperiments.mat")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "frequency_domain"
CHANNELS = range(1, 21)
CONVERT_TO_MS2 = False               # False: g; True: m/s^2.
FFT_START_SECONDS = 0.0
FFT_END_SECONDS = 660.0              # Analyze the same interval as time plots.
REMOVE_MEAN = True
WINDOW = "hann"                     # "hann" or "none".
MAX_FREQUENCY_HZ = None              # None: full spectrum; e.g. 50.0 to zoom.
SHOW_PLOTS = True
# -------------------------------------------------------------------------


def compute_fft(time_seconds, acceleration, sampling_rate_hz,
                remove_mean=True, window="hann"):
    """Return frequencies and one-sided peak amplitudes for uniform samples.

    Pass the desired time slice. DC and the even-length Nyquist bin are not
    doubled. No filtering, resampling, or zero padding is performed.
    """
    t = np.asarray(time_seconds, dtype=float)
    a = np.asarray(acceleration, dtype=float)
    if t.ndim != 1 or a.ndim != 1 or t.size != a.size or t.size < 4:
        raise ValueError("Time and acceleration must be matching 1D arrays with at least four samples.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(a)):
        raise ValueError("Time and acceleration must contain only finite values.")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("Sampling rate must be finite and positive.")
    steps = np.diff(t)
    if not np.allclose(steps, 1.0 / sampling_rate_hz, rtol=1e-4, atol=1e-10):
        raise ValueError("Time samples must be uniformly spaced and match the sampling rate.")

    if window == "hann":
        weights = hann(a.size, sym=False)
    elif window == "none":
        weights = np.ones(a.size)
    else:
        raise ValueError('WINDOW must be "hann" or "none".')

    signal = a - np.mean(a) if remove_mean else a
    amplitude = np.abs(rfft(signal * weights)) / weights.sum()
    if a.size % 2 == 0:
        amplitude[1:-1] *= 2.0
    else:
        amplitude[1:] *= 2.0
    frequency_hz = rfftfreq(a.size, d=1.0 / sampling_rate_hz)
    return frequency_hz, amplitude


def main():
    if not (np.isfinite(FFT_START_SECONDS) and np.isfinite(FFT_END_SECONDS)
            and FFT_END_SECONDS > FFT_START_SECONDS):
        raise ValueError("FFT end time must be finite and greater than the start time.")
    if MAX_FREQUENCY_HZ is not None and (
            not np.isfinite(MAX_FREQUENCY_HZ) or MAX_FREQUENCY_HZ <= 0):
        raise ValueError("MAX_FREQUENCY_HZ must be positive or None.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for channel in CHANNELS:
        print(f"Loading channel {channel}...", flush=True)
        t, a, metadata = load_channel(MAT_FILE, channel, convert_to_ms2=CONVERT_TO_MS2)
        # Copy the interval so the full channel can be released before the FFT.
        first = np.searchsorted(t, FFT_START_SECONDS, side="left")
        last = np.searchsorted(t, FFT_END_SECONDS, side="right")
        selected_t = t[first:last].copy()
        selected_a = a[first:last].copy()
        del t, a
        try:
            frequencies, amplitudes = compute_fft(
                selected_t, selected_a, metadata["sampling_rate_hz"],
                remove_mean=REMOVE_MEAN, window=WINDOW,
            )
        except ValueError as error:
            raise ValueError(f"Channel {channel}: {error}") from error

        nyquist = metadata["sampling_rate_hz"] / 2.0
        frequency_limit = nyquist if MAX_FREQUENCY_HZ is None else min(MAX_FREQUENCY_HZ, nyquist)
        visible = frequencies <= frequency_limit
        fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        ax.plot(frequencies[visible], amplitudes[visible], linewidth=0.7)
        ax.set(
            xlabel="Frequency (Hz)",
            ylabel=f"Peak acceleration amplitude ({metadata['acceleration_unit']})",
            title=(f"Walking experiment - Channel {channel} FFT\n"
                   f"{selected_t[0]:g}-{selected_t[-1]:g} s | "
                   f"Window: {WINDOW} | Mean removed: {REMOVE_MEAN}"),
            xlim=(0.0, frequency_limit),
            ylim=(0.0, None),
        )
        ax.grid(True, alpha=0.3)
        output_path = OUTPUT_DIR / f"channel_{channel:02d}_fft.png"
        fig.savefig(output_path, dpi=300)
        print(f"  {selected_t.size:,} samples; fs = {metadata['sampling_rate_hz']:g} Hz; "
              f"frequency spacing = {frequencies[1]:.6g} Hz", flush=True)
        print(f"  Saved: {output_path}", flush=True)
        if not SHOW_PLOTS:
            plt.close(fig)
        del selected_t, selected_a, frequencies, amplitudes

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
