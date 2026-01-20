"""
EEG Signal Processing and Feature Extraction Module

This module provides a comprehensive EEG processing pipeline including:
- Data loading from Parquet/CSV files
- Signal filtering and resampling
- Power band decomposition (delta, theta, alpha, beta, gamma)
- Feature extraction (entropy, ZCR, brain rate, statistical features)
- GPU acceleration support
- Visualization capabilities

Key Features:
1. Automated EEG channel name standardization
2. GPU-accelerated signal processing (when available)
3. Time-domain and frequency-domain feature extraction
4. Sliding window analysis capabilities
5. Comprehensive statistical feature generation

Typical workflow:
1. Initialize EEG object from file
2. Apply basic preprocessing (filtering, normalization)
3. Decompose into frequency bands
4. Extract statistical features
5. Export results

Dependencies: pandas, numpy, scipy, matplotlib, mne, spkit, EntropyHub
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy.signal import resample_poly
import scipy.io as sio
import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Optional
import mne
from mne.filter import filter_data
import spkit as sp
from EntropyHub import FuzzEn
from math import ceil
import re
from fractions import Fraction

# GPU support detection - falls back to CPU if unavailable
try:
    import cupy as cp
    from cupyx.scipy.signal import sosfilt as gpu_sosfilt
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np


class EEG:
    """
    EEG Signal Processing Core Class

    Handles loading, processing, and feature extraction from EEG data.

    Attributes:
        data (pd.DataFrame): Preprocessed EEG data with multi-index columns (band, channel)
        t (np.ndarray): Timestamps in seconds
        frequency (float): Sampling frequency in Hz
        channels (list): Valid EEG channel names
        band_powers (pd.DataFrame): Absolute/relative power per band
        stats (pd.DataFrame): Comprehensive feature matrix

    Example:
        eeg = EEG.from_file('data.parquet', frequency=256, time_col='timestamp')
        eeg.generate_stats()
        eeg.export_metrics('features.parquet')
    """

    def __init__(
        self,
        s_n: np.ndarray,
        t: np.ndarray,
        channels: dict[str, np.ndarray],
        frequency: float,
        extract_time: bool = False
    ):
        """
        Initialize EEG processor.

        Args:
            s_n: Sample numbers
            t: Timestamps (any format convertible to numeric seconds)
            channels: Dictionary of {channel_name: data_array}
            frequency: Sampling frequency in Hz
            extract_time: Whether to extract time features (not implemented)
        """
        # Store raw data and convert timestamps to seconds
        self.data = pd.DataFrame({**channels})
        try:
            self.t = pd.to_numeric(pd.to_datetime(t).astype('int64')) / 1e9  # Convert to seconds
        except Exception:
            self.t = pd.to_numeric(t)
        
        # Standardize channel names (Fp1, Fp2, etc.)
        channel_map = {orig: self._renamer(orig) for orig in self.data.columns}
        self.data.rename(mapper=channel_map, axis=1, inplace=True)
        self.channels = list(self.data.columns)
        self.frequency = frequency
        
        # Configure GPU/CPU processing
        self._setup_acceleration()
        
        # Core processing pipeline
        self.data = self.basic_filter()  # 0.5-45Hz bandpass
        self.data.columns = pd.MultiIndex.from_product([["Overall"], self.data.columns])
        
        # Resample AFTER MultiIndex is created (if needed)
        if extract_time:
            self.resample()
        
        # Frequency band decomposition
        band_waves, self.band_powers = self.accelerated_power_bands()
        self.data = self.data.join(band_waves)
        
        # Normalization and final structure
        self.data = self.normalise_eeg()
        self.data = self.data[[col for col in self.data.columns if isinstance(col, tuple)]]
        self.data.columns = pd.MultiIndex.from_tuples(
            self.data.columns, 
            names=["band", "channel"]
        )

    # --------------------------
    # Core Processing Methods
    # --------------------------
    
    def generate_stats(self):
        """Compute comprehensive feature set including:
        - Band power ratios
        - Peak statistics
        - Time-domain statistics
        - Entropy measures
        - Brain rate index
        """
        self.ratio_indices = self.calc_ratio_indices(powers=self.band_powers)
        self.num_peaks = self.calc_num_peaks()
        self.channel_stats = self.calc_channel_stats(
            ["mean", "median", "var", "std", "skew", "kurt"]
        )
        self.zcr = self.calc_zcr()
        self.renyi_entropy = self.calc_renyi_entropy()
        self.diff_entropy = self.calc_diff_entropy()
        self.fuzzy_entropy = self.calc_fuzzy_entropy()
        self.overall_brain_rate = self.calc_brain_rate_overall()
        
        # Combine all features into single dataframe
        self.stats = pd.concat([
            self.band_powers, self.ratio_indices, self.num_peaks,
            self.channel_stats, self.zcr, self.renyi_entropy,
            self.diff_entropy, self.fuzzy_entropy, self.overall_brain_rate
        ], axis=0)

    def generate_time_data(self, window_len: float = 2.0, window_step: float = 0):
        """
        Prepare for sliding window analysis (call before export_time_data_windows).
        
        Args:
            window_len: Window length in seconds
            window_step: Step size in seconds (default = window_len)
        """
        self.window_len = window_len
        self.window_step = window_step or self.window_len

    # --------------------------
    # Initialization Helpers
    # --------------------------

    def _setup_acceleration(self):
        """Configure GPU resources if available"""
        self.use_gpu = GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.lfilter = gpu_sosfilt if self.use_gpu else signal.sosfilt

    @classmethod
    def from_file(
        cls, 
        file_path: str, 
        frequency: float, 
        time_col: Optional[str],
        channel_cols: Optional[list[str]] = None,
        extract_time: bool = False,
        filetype: Optional[str] = "parquet"
    ):
        """
        Create EEG instance from data file.

        Args:
            file_path: Path to data file
            frequency: Sampling frequency in Hz
            time_col: Name of timestamp column
            channel_cols: Specific EEG channels to load
            extract_time: Whether to extract time features
            filetype: 'parquet' or 'csv'

        Returns:
            EEG instance
        """
        if filetype == "parquet":
            data = pd.read_parquet(file_path)
        elif filetype == "csv":
            data = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported filetype")
            
        if time_col is not None:
            data.rename(columns={time_col: "t"}, inplace=True)
            
        if channel_cols is not None:
            columns = ["t"] + channel_cols
        else:
            columns = data.columns.to_list()
            
        # Add sample numbers if missing
        if "s_n" not in columns:
            data['s_n'] = np.arange(len(data)) + 1
            
        return cls(
            data['s_n'].values,
            data['t'].values,
            {col: data[col].values for col in columns if col not in ('t', 's_n')},
            frequency,
            extract_time
        )

    def _renamer(self, ch: str) -> str:
        """Standardize channel names to canonical form (e.g., 'EEG_Fp1' -> 'Fp1')"""
        # Remove prefixes
        ch = re.sub(r'^[^A-Za-z0-9]*(EEG[_\-\s.]*)?', '', ch, flags=re.IGNORECASE)
        # Remove suffixes
        ch = re.sub(r'(-Pz|-Avg|_REF| EEG)$', '', ch, flags=re.IGNORECASE)
        # Format case
        return ch.strip().lower().capitalize()

    # --------------------------
    # Signal Processing Methods
    # --------------------------

    def resample(self, target_freq: int = 128):
        """
        Resample data to target frequency (currently downsampling only).
        
        Args:
            target_freq: Target sampling frequency in Hz
        """
        if self.frequency <= target_freq:
            raise NotImplementedError("Upsampling not implemented")
            
        channels = self.channels
        resampled_channels = {}
        
        # Calculate resampling ratio
        frac = Fraction(self.frequency, target_freq).limit_denominator(1000)
        up, down = frac.denominator, frac.numerator

        for ch in channels:
            signal = self.data[("Overall", ch)].values
            resampled = resample_poly(signal, up, down)
            resampled_channels[ch] = resampled

        # Update metadata
        self.frequency = target_freq
        self.data = pd.DataFrame(resampled_channels)
        self.data.columns = pd.MultiIndex.from_product([["Overall"], self.data.columns])

    def basic_filter(self, channels: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Apply 0.5-45Hz bandpass filter (IIR Butterworth 4th order).
        
        Args:
            channels: Specific channels to filter (default: all)
            
        Returns:
            Filtered DataFrame
        """
        channels = self._process_channels_input(channels)
        data = self.data[channels].values.T
        nyq = 0.5 * self.frequency
        
        # Design filter
        low = 0.5 / nyq
        high = 45 / nyq
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        
        # Apply filter (GPU/CPU)
        if self.use_gpu:
            data_gpu = cp.asarray(data, dtype=cp.float64)
            sos_gpu = cp.asarray(sos)
            filtered = cp.asnumpy(gpu_sosfilt(sos_gpu, data_gpu))
        else:
            filtered = signal.sosfilt(sos, data, axis=1)
            
        return pd.DataFrame(filtered.T, columns=channels)

    def accelerated_power_bands(self, nperseg: int = 256) -> tuple:
        """
        Calculate band power features using Welch's method.
        
        Args:
            nperseg: FFT segment length
            
        Returns:
            band_waves: Band-pass filtered signals
            band_powers: Power features (absolute, relative)
        """
        bands = {
            "delta": (0.5, 4), "theta": (4, 8),
            "alpha": (8, 13), "beta": (13, 30),
            "gamma": (30, 45)
        }

        # Adjust segment length if too long
        if nperseg > len(self.data):
            nperseg = len(self.data) // 4
        
        # Total power for normalization
        min_freq = 0.5
        max_freq = 45
        total_powers = self._calculate_power(min_freq, max_freq, nperseg)

        band_waves_list = []
        power_data = []

        for band, (lowcut, highcut) in bands.items():
            # Band-pass filtered signals
            filtered_data = self._get_band_data(lowcut, highcut)
            band_df = pd.DataFrame(
                filtered_data,
                columns=pd.MultiIndex.from_product([[band], self.channels]),
                index=self.data.index
            )
            band_waves_list.append(band_df)

            # Power calculations
            abs_power = self._calculate_power(lowcut, highcut, nperseg)
            rel_power = abs_power / total_powers
            rel_power = np.clip(rel_power, 0, 1)

            # Store metrics
            df = pd.DataFrame({
                'absolute_power': abs_power,
                'relative_power': rel_power
            }, index=self.channels)
            df['band'] = band
            power_data.append(df)

        # Combine results
        band_waves = pd.concat(band_waves_list, axis=1)
        power_df = pd.concat(power_data)
        power_df = power_df.set_index('band', append=True).unstack(0).stack(0)
        power_df.columns.names = ['metric', 'channel']
        
        return band_waves, power_df

    def _calculate_power(self, lowcut: float, highcut: float, nperseg: int) -> np.ndarray:
        """Compute power in frequency band using Welch's PSD (CPU & GPU equivalent)."""
        raw_data = self.data["Overall"][self.channels].values.T.astype(np.float64)

        if self.use_gpu:
            data_gpu = cp.asarray(raw_data)
            _, psd_gpu = self._gpu_welch(data_gpu, lowcut, highcut, nperseg)
            psd_sum = cp.asnumpy(psd_gpu.sum(axis=1))
        else:
            # Manual CPU implementation mirroring _gpu_welch
            n_channels, n_samples = raw_data.shape
            fs = self.frequency

            f = np.fft.rfftfreq(nperseg, 1 / fs)
            window = np.hanning(nperseg)
            norm_factor = np.sum(window) ** 2  # same as GPU
            step = nperseg // 4                # 75% overlap (move 25% each step)
            psd = np.zeros((n_channels, len(f)), dtype=np.float64)
            n_segments = 0

            for start in range(0, n_samples - nperseg, step):
                seg = raw_data[:, start:start + nperseg]
                seg = seg - np.mean(seg, axis=1, keepdims=True)
                seg_windowed = seg * window
                fft_seg = np.fft.rfft(seg_windowed, axis=1)
                psd += np.abs(fft_seg) ** 2 / norm_factor
                n_segments += 1

            psd /= max(1, n_segments)
            mask = (f >= lowcut) & (f <= highcut)
            psd_sum = psd[:, mask].sum(axis=1)

        # Numerical stability
        return np.clip(psd_sum, 1e-12, None)


    def _gpu_welch(self, data, low: float, high: float, nperseg: int) -> tuple:
        """GPU implementation of Welch's method with 75% overlap
        changes made on 10/
        
        """
        n_channels, n_samples = data.shape
        f_gpu = cp.fft.rfftfreq(nperseg, 1/self.frequency)
        psd_gpu = cp.zeros((n_channels, len(f_gpu)), dtype=cp.float64)

        # Windowed FFT with overlap
        window = cp.hanning(nperseg).astype(cp.float64)
        norm_factor = cp.sum(window)**2
        
        for i in range(0, n_samples - nperseg, nperseg//4):
            seg = data[:, i:i+nperseg].astype(cp.float64)
            windowed_seg = seg * window
            fft = cp.fft.rfft(windowed_seg, axis=1)
            psd_gpu += cp.abs(fft)**2 / norm_factor

        psd_gpu /= (n_samples // (nperseg//4))  # Overlap compensation
        
        # Frequency masking
        mask = (f_gpu >= low) & (f_gpu <= high)
        return f_gpu[mask], psd_gpu[:, mask]

    def _get_band_data(self, lowcut: float, highcut: float) -> np.ndarray:
        """Bandpass filter data for specific frequency band"""
        nyq = self.frequency * 0.5
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(4, [low, high], btype='band', output='sos')

        if self.use_gpu:
            data_gpu = cp.asarray(self.data["Overall"][self.channels].values.T)
            sos_gpu = cp.asarray(sos)
            return cp.asnumpy(gpu_sosfilt(sos_gpu, data_gpu)).T
        else:
            return signal.sosfilt(
                sos,
                self.data["Overall"][self.channels].values.T,
                axis=1
            ).T

    # --------------------------
    # Feature Extraction Methods
    # --------------------------

    def calc_ratio_indices(
        self, 
        powers: Optional[pd.DataFrame] = None, 
        channels: Optional[list[str]] = None,
        eps: float = 1e-12
    ) -> pd.DataFrame:
        """
        Calculate power band ratios:
        1. Theta/Beta
        2. (Theta+Alpha)/(Alpha+Beta)
        3. Gamma/Delta
        4. (Gamma+Beta)/(Delta+Alpha)
        
        Args:
            powers: Precomputed power DataFrame
            channels: Specific channels to process
            eps: Small value to prevent division by zero
            
        Returns:
            DataFrame of ratio indices
        """
        channels = self._process_channels_input(channels)
        powers = powers or self.band_powers
        
        indices = [
            "theta/beta", "(theta + alpha)/(alpha + beta)",
            "gamma/delta", "(gamma + beta)/(delta + alpha)"
        ]
        
        ratio_data = []
        for channel in channels:
            # Extract band powers
            theta = powers.loc[('absolute_power', channel), 'theta']
            beta = powers.loc[('absolute_power', channel), 'beta']
            alpha = powers.loc[('absolute_power', channel), 'alpha']
            gamma = powers.loc[('absolute_power', channel), 'gamma']
            delta = powers.loc[('absolute_power', channel), 'delta']

            # Compute ratios with safeguards
            ratios = [
                theta / max(beta, eps),
                (theta + alpha) / max(alpha + beta, eps),
                gamma / max(delta, eps),
                (gamma + beta) / max(delta + alpha, eps)
            ]
            ratio_data.append(ratios)
        
        return pd.DataFrame(
            ratio_data,
            index=channels,
            columns=indices
        ).T

    def calc_num_peaks(
        self,
        percentile: float = 95,
        min_height: Optional[float] = None,
        prominence: float = 1,
        distance: Optional[float] = None,
        channels: Optional[list[str]] = None,
        bands: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate peak density per band-channel.
        
        Args:
            percentile: Height threshold percentile
            min_height: Absolute minimum peak height
            prominence: Minimum peak prominence
            distance: Minimum peak spacing (samples)
            channels: Specific channels to process
            bands: Specific bands to process
            
        Returns:
            DataFrame of peak densities
        """
        channels = self._process_channels_input(channels)
        bands = bands or self.data.columns.get_level_values(0).unique().tolist()
        
        peak_counts = []
        for band in bands:
            for channel in channels:
                series = self.data[(band, channel)]
                # Calculate dynamic threshold
                threshold = np.percentile(series, percentile)
                height = max(min_height, threshold) if min_height else threshold
                
                # Detect peaks
                peaks, _ = signal.find_peaks(
                    series,
                    height=height,
                    prominence=prominence,
                    distance=distance
                )
                
                # Normalize by signal length
                peak_counts.append({
                    "band": band,
                    "channel": channel,
                    "num_peaks": len(peaks) / len(series)
                })
        
        return pd.DataFrame(peak_counts).groupby(
            ["band", "channel"]
        )["num_peaks"].mean().to_frame(name="num_peaks").T

    def calc_channel_stats(
        self, 
        stats: list[str],
        channels: Optional[list[str]] = None,
        bands: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Compute statistical moments for each band-channel.
        
        Args:
            stats: List of statistics ('mean', 'var', etc.)
            channels: Specific channels to process
            bands: Specific bands to process
            
        Returns:
            DataFrame with statistics as rows, band-channels as columns
        """
        channels = self._process_channels_input(channels)
        bands = bands or self.data.columns.get_level_values(0).unique().tolist()
        
        # Build column selector
        columns = [(band, ch) for band in bands for ch in channels]
        subset = self.data[columns]
        
        # Compute statistics
        return subset.agg(stats)

    def normalise_eeg(self, channels: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Z-score normalization per band-channel.
        
        Args:
            channels: Specific channels to normalize
            
        Returns:
            Normalized DataFrame
        """
        channels = self._process_channels_input(channels)
        bands = self.data.columns.get_level_values(0).unique().tolist()
        
        columns = [(band, ch) for band in bands for ch in channels]
        data = self.data[columns].values
        
        # Standard scaling
        normalized = (data - data.mean(axis=0)) / data.std(axis=0)
        return pd.DataFrame(normalized, columns=columns)

    def calc_zcr(
        self, 
        channels: Optional[list[str]] = None,
        bands: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Compute Zero Crossing Rate (ZCR).
        
        Args:
            channels: Specific channels to process
            bands: Specific bands to process
            
        Returns:
            DataFrame of ZCR values
        """
        channels = self._process_channels_input(channels)
        bands = bands or self.data.columns.get_level_values(0).unique().tolist()
        
        columns = [(band, ch) for band in bands for ch in channels]
        data = self.data[columns].T.values
        
        # Compute sign changes
        signs = np.sign(data)
        diffs = np.diff(signs, axis=1)
        crossings = np.count_nonzero(diffs != 0, axis=1)
        zcr_values = crossings / (data.shape[1] - 1)
        
        return pd.DataFrame([zcr_values], columns=columns, index=["zcr"])

    def calc_renyi_entropy(self, order: float = 2) -> pd.DataFrame:
        """Compute Renyi entropy (order=2 by default) per band-channel"""
        return self._entropy_wrapper("renyi_entropy", sp.entropy, alpha=order)

    def calc_diff_entropy(self) -> pd.DataFrame:
        """Compute differential entropy per band-channel"""
        return self._entropy_wrapper("diff_entropy", stats.differential_entropy)

    def _entropy_wrapper(self, name: str, func: callable, **kwargs) -> pd.DataFrame:
        """Generic entropy calculation helper"""
        bands = self.data.columns.get_level_values(0).unique()
        entropy_vals = []
        indices = []
        
        for band in bands:
            for channel in self.channels:
                series = self.data[(band, channel)].values
                entropy_vals.append(func(series, **kwargs))
                indices.append((band, channel))
                
        return pd.DataFrame(
            entropy_vals,
            index=pd.MultiIndex.from_tuples(indices),
            columns=[name]
        ).T

    def calc_fuzzy_entropy(self, window_size: int = 20) -> pd.DataFrame:
        """
        Compute fuzzy entropy per band-channel.
        
        Args:
            window_size: Embedding dimension for entropy calculation
            
        Returns:
            DataFrame of entropy values (dim1 and dim2)
        """
        bands = self.data.columns.get_level_values(0).unique().tolist()
        entropy_vals = []
        indices = []
        
        for band in bands:
            for ch in self.channels:
                data = self.data[(band, ch)].values.squeeze()
                if len(data) < window_size:
                    entropy_vals.append([np.nan, np.nan])
                else:
                    # Segment data into windows
                    n_windows = len(data) // window_size
                    truncated = data[:n_windows * window_size]
                    windows = truncated.reshape(n_windows, window_size)
                    
                    # Compute entropy per window
                    entropies = [FuzzEn(w)[0] for w in windows]
                    entropy_vals.append(np.nanmean(entropies, axis=0))
                indices.append((band, ch))
                
        return pd.DataFrame(
            entropy_vals,
            index=pd.MultiIndex.from_tuples(indices),
            columns=["fuzzy_entropy_dim1", "fuzzy_entropy_dim2"]
        ).T

    def calc_brain_rate_overall(self) -> pd.DataFrame:
        """
        Compute overall brain rate index using:
        BR = Σ (RelativeBandPower × BandWeight)
        
        Weights: delta=2.25, theta=6.0, alpha=10.5, beta=21.5, gamma=37.5
        
        Returns:
            DataFrame with brain rate per channel
        """
        band_weights = {
            "delta": 2.25, "theta": 6.0,
            "alpha": 10.5, "beta": 21.5,
            "gamma": 37.5
        }
        
        # Calculate relative amplitudes
        rel_amplitudes = self.band_powers.loc['relative_power']
        
        # Apply weights and sum
        weighted = rel_amplitudes.mul(
            pd.Series(band_weights), 
            axis=0, 
            level=0
        )
        brain_rate = weighted.groupby(level=1).sum()
        
        # Format results
        return pd.DataFrame(
            [brain_rate.values],
            columns=pd.MultiIndex.from_product([["Overall"], brain_rate.index]),
            index=["brain_rate"]
        )

    # --------------------------
    # Utility Methods
    # --------------------------

    def _process_channels_input(self, channels: Optional[list[str]] = None) -> list[str]:
        """Validate channel list or return all channels"""
        if channels is None:
            return self.channels
        invalid = [ch for ch in channels if ch not in self.channels]
        if invalid:
            raise ValueError(f"Invalid channels: {', '.join(invalid)}")
        return channels

    def plot_channels(
        self, 
        band: str,
        channels: Optional[list[str]] = None
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot EEG traces for specified channels.
        
        Args:
            band: Frequency band to plot ('Overall', 'alpha', etc.)
            channels: Channels to include
            
        Returns:
            Matplotlib figure and axes
        """
        channels = self._process_channels_input(channels)
        
        n_plots = len(channels)
        n_rows = (n_plots + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3*n_rows))
        fig.suptitle(f'{band} Band EEG Traces', fontsize=16)
        
        for i, ch in enumerate(channels):
            ax = axes.flat[i]
            ax.plot(self.t, self.data[(band, ch)])
            ax.set_title(ch, fontsize=12)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (μV)")
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig, axes

    # --------------------------
    # Export Methods
    # --------------------------

    def export_time_data(self, file_path: str):
        """Export full time-series data to Parquet"""
        self.data.to_parquet(file_path)

    def export_time_data_windows(self, file_path: str):
        """Export data in sliding window chunks"""
        window_samples = int(self.frequency * self.window_len)
        step_samples = int(self.frequency * self.window_step)
        
        for i, start in enumerate(range(0, len(self.data) - window_samples + 1, step_samples)):
            window = self.data.iloc[start:start+window_samples]
            window.to_parquet(f"{file_path}_window{i}.parquet")

    def export_metrics(
        self, 
        file_name: str, 
        metrics: Optional[list] = None, 
        bands: Optional[list] = None,
        channels: Optional[list] = None
    ):
        """Export feature matrix subset to Parquet"""
        if metrics is None:
            metrics = self.stats.index
        if bands is None:
            bands = self.stats.columns.get_level_values(0).unique()
        if channels is None:
            channels = self.stats.columns.get_level_values(1).unique()
            
        # Advanced multi-index slicing
        idx = pd.IndexSlice
        subset = self.stats.loc[
            metrics,
            idx[bands, channels]
        ]
        subset.to_parquet(file_name)