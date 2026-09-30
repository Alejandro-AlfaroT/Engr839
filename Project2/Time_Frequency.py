"""Create unfiltered STFT spectrograms for all walking experiment channels.

Run in PyCharm, or with: python Time_Frequency.py
Requires numpy, scipy (with ShortTimeFFT), matplotlib, and Data_Import.py.
Saves outputs/time_frequency/raw/channel_XX_spectrogram.png at 300 DPI.

Each figure shows the full frequency range and a 700-900 Hz detail. Colors
represent one-sided power spectral density (PSD), not FFT peak amplitude.
Raw means no frequency filtering or resampling; by default each window's mean
is removed before applying a periodic Hann window. Only complete windows are
used, so there are no padded boundary frames. Times mark window centers.
Fixed color limits allow comparisons between channels and future filtered plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from Data_Import import load_channel


# -------------------------- EDIT THESE SETTINGS ---------------------------
MAT_FILE = Path(r"D:\839WalkingExperiments.mat")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "time_frequency" / "raw"
CHANNELS = range(1, 21)
CONVERT_TO_MS2 = False               # False: g; True: m/s^2.
START_SECONDS = 0.0
END_SECONDS = 660.0
WINDOW_SECONDS = 2.0
OVERLAP_FRACTION = 0.75
REMOVE_WINDOW_MEAN = True
DETAIL_BAND_HZ = (700.0, 900.0)
PSD_MIN_DB = -160.0                  # Fixed limits in dB re 1 g^2/Hz.
PSD_MAX_DB = -40.0                   # Automatically shifted if using m/s^2.
SHOW_PLOTS = True
# -------------------------------------------------------------------------


def compute_spectrogram(time_seconds, acceleration, sampling_rate_hz,
                        window_seconds=2.0, overlap_fraction=0.75,
                        remove_mean=True):
    """Return (frequency_hz, absolute_center_times, one_sided_psd, hop_seconds).

    PSD units are acceleration_unit^2/Hz. No zero padding is used in the FFT.
    With 2560 Hz sampling, defaults give 0.5 Hz bins and a 0.5 s hop; the
    window still spans 2 s, and neighboring frames overlap.
    """
    t = np.asarray(time_seconds, dtype=float)
    a = np.asarray(acceleration, dtype=float)
    if t.ndim != 1 or a.ndim != 1 or t.size != a.size or t.size < 4:
        raise ValueError("Time and acceleration must be matching 1D arrays with at least four samples.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(a)):
        raise ValueError("Time and acceleration must contain only finite values.")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("Sampling rate must be finite and positive.")
    if not np.allclose(np.diff(t), 1.0 / sampling_rate_hz, rtol=1e-4, atol=1e-10):
        raise ValueError("Time samples must be uniformly spaced and match the sampling rate.")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Window duration must be finite and positive.")
    if not np.isfinite(overlap_fraction) or not 0 <= overlap_fraction < 1:
        raise ValueError("Overlap fraction must be at least zero and less than one.")
    window_samples = round(window_seconds * sampling_rate_hz)
    if window_samples < 4 or a.size < window_samples:
        raise ValueError("The window must have at least four samples and fit inside the selected interval.")
    hop = max(1, round(window_samples * (1 - overlap_fraction)))
    transform = ShortTimeFFT(
        hann(window_samples, sym=False), hop=hop, fs=sampling_rate_hz,
        fft_mode="onesided2X", scale_to="psd",
    )
    # Center the first frame on half a window and stop before any padded frame.
    frame_count = 1 + (a.size - window_samples) // hop
    center_offset = window_samples // 2
    psd = transform.spectrogram(
        a, detr="constant" if remove_mean else None,
        p0=0, p1=frame_count, k_offset=center_offset,
    )
    times = t[0] + transform.t(a.size, p0=0, p1=frame_count, k_offset=center_offset)
    return transform.f, times, psd, hop / sampling_rate_hz


def main():
    if not (np.isfinite(START_SECONDS) and np.isfinite(END_SECONDS)
            and END_SECONDS > START_SECONDS):
        raise ValueError("End time must be finite and greater than the start time.")
    if not (np.isfinite(PSD_MIN_DB) and np.isfinite(PSD_MAX_DB) and PSD_MAX_DB > PSD_MIN_DB):
        raise ValueError("PSD color limits must be finite and increasing.")
    if not (len(DETAIL_BAND_HZ) == 2 and np.all(np.isfinite(DETAIL_BAND_HZ))
            and 0 <= DETAIL_BAND_HZ[0] < DETAIL_BAND_HZ[1]):
        raise ValueError("The detail frequency band must have increasing nonnegative limits.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unit_db_shift = 20 * np.log10(9.80665) if CONVERT_TO_MS2 else 0.0

    for channel in CHANNELS:
        print(f"Loading channel {channel}...", flush=True)
        t, a, metadata = load_channel(MAT_FILE, channel, convert_to_ms2=CONVERT_TO_MS2)
        first = np.searchsorted(t, START_SECONDS, side="left")
        last = np.searchsorted(t, END_SECONDS, side="right")
        selected_t, selected_a = t[first:last].copy(), a[first:last].copy()
        del t, a
        try:
            frequencies, times, psd, hop_seconds = compute_spectrogram(
                selected_t, selected_a, metadata["sampling_rate_hz"],
                WINDOW_SECONDS, OVERLAP_FRACTION, REMOVE_WINDOW_MEAN,
            )
        except ValueError as error:
            raise ValueError(f"Channel {channel}: {error}") from error
        nyquist = metadata["sampling_rate_hz"] / 2
        if DETAIL_BAND_HZ[0] >= nyquist:
            raise ValueError(f"Channel {channel}: detail frequency band is above Nyquist ({nyquist:g} Hz).")
        db = 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))
        df = frequencies[1] - frequencies[0]
        extent = (times[0] - hop_seconds / 2, times[-1] + hop_seconds / 2,
                  frequencies[0] - df / 2, frequencies[-1] + df / 2)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        for ax, limits, title in zip(
                axes, [(0, nyquist), (DETAIL_BAND_HZ[0], min(DETAIL_BAND_HZ[1], nyquist))],
                ["Full frequency range", "High-frequency detail"]):
            mesh = ax.imshow(
                db, origin="lower", aspect="auto", extent=extent,
                interpolation="nearest", cmap="magma",
                vmin=PSD_MIN_DB + unit_db_shift, vmax=PSD_MAX_DB + unit_db_shift,
            )
            ax.set(ylabel="Frequency (Hz)", ylim=limits,
                   xlim=(START_SECONDS, END_SECONDS), title=title)
        axes[-1].set_xlabel("Time (s); window centers")
        unit = metadata["acceleration_unit"]
        psd_unit = "g^2/Hz" if unit == "g" else "(m/s^2)^2/Hz"
        fig.colorbar(mesh, ax=axes, label=f"PSD (dB re 1 {psd_unit})", extend="both")
        fig.suptitle(
            f"Walking experiment - Channel {channel} | Unfiltered spectrogram\n"
            f"Hann: {WINDOW_SECONDS:g} s | Overlap: {OVERLAP_FRACTION:.0%} | "
            f"Window mean removed: {REMOVE_WINDOW_MEAN}"
        )
        output_path = OUTPUT_DIR / f"channel_{channel:02d}_spectrogram.png"
        fig.savefig(output_path, dpi=300)
        print(f"  {times.size} complete windows; {df:g} Hz bins; {hop_seconds:g} s hop", flush=True)
        print(f"  Saved: {output_path}", flush=True)
        if not SHOW_PLOTS:
            plt.close(fig)
        del selected_t, selected_a, frequencies, times, psd, db

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
