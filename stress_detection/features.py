import mne_features.univariate as mne_f
import numpy as np
from scipy.signal import welch

def time_series_features(data):
    '''
    Computes the features variance, RMS and peak-to-peak amplitude using the package mne_features.

    Args:
        data (ndarray): EEG data.

    Returns:
        ndarray: Computed features.

    '''

    n_trials, n_secs, n_channels, _ = data.shape
    features_per_channel = 3

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            variance = mne_f.compute_variance(second)
            rms = mne_f.compute_rms(second)
            ptp_amp = mne_f.compute_ptp_amp(second)
            features[i][j] = np.concatenate([variance, rms, ptp_amp])
    features = features.reshape(
        [n_trials*n_secs, n_channels*features_per_channel])
    return features


def freq_band_features(data, freq_bands):
    '''
    Computes the frequency bands delta, theta, alpha, beta and gamma using the package mne_features.

    Args:
        data (ndarray): EEG data.
        freq_bands (ndarray): The frequency bands to compute.

    Returns:
        ndarray: Computed features.
    '''
    n_trials, n_secs, n_channels, sfreq = data.shape
    features_per_channel = len(freq_bands)-1

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            psd = mne_f.compute_pow_freq_bands(
                sfreq, second, freq_bands=freq_bands)
            features[i][j] = psd
    features = features.reshape(
        [n_trials*n_secs, n_channels*features_per_channel])
    return features


def hjorth_features(data):
    '''
    Computes the features Hjorth mobility (spectral) and Hjorth complexity (spectral) using the package mne_features.

    Args:
        data (ndarray): EEG data.

    Returns:
        ndarray: Computed features.
    '''
    n_trials, n_secs, n_channels, sfreq = data.shape
    features_per_channel = 2

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            mobility_spect = mne_f.compute_hjorth_mobility_spect(sfreq, second)
            complexity_spect = mne_f.compute_hjorth_complexity_spect(
                sfreq, second)
            features[i][j] = np.concatenate([mobility_spect, complexity_spect])
    features = features.reshape(
        [n_trials*n_secs, n_channels*features_per_channel])
    return features


def fractal_features(data):
    '''
    Computes the Higuchi Fractal Dimension and Katz Fractal Dimension using the package mne_features.

    Args:
        data (ndarray): EEG data.

    Returns:
        ndarray: Computed features.

    '''
    n_trials, n_secs, n_channels, _ = data.shape
    features_per_channel = 2

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            higuchi = mne_f.compute_higuchi_fd(second)
            katz = mne_f.compute_katz_fd(second)
            features[i][j] = np.concatenate([higuchi, katz])
    features = features.reshape(
        [n_trials*n_secs, n_channels*features_per_channel])
    return features


def entropy_features(data):
    '''
    Computes the features Approximate Entropy, Sample Entropy, Spectral Entropy and SVD entropy using the package mne_features.

    Args:
        data (ndarray): EEG data.

    Returns:
        ndarray: Computed features.

    '''
    n_trials, n_secs, n_channels, sfreq = data.shape
    features_per_channel = 4

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            app_entropy = mne_f.compute_app_entropy(second)
            samp_entropy = mne_f.compute_samp_entropy(second)
            spect_entropy = mne_f.compute_spect_entropy(sfreq, second)
            svd_entropy = mne_f.compute_svd_entropy(second)
            features[i][j] = np.concatenate(
                [app_entropy, samp_entropy, spect_entropy, svd_entropy])
    features = features.reshape(
        [n_trials*n_secs, n_channels*features_per_channel])
    return features


def compute_band_power_welch(segment, sfreq, eps=1e-8):
    '''
    Computes EPOC X aligned band power features from EEG segment.

    Args:
        segment: EEG segment with shape (n_channels, n_samples).
        sfreq: Sampling frequency.
        eps: Small value to avoid numerical problems.

    Returns:
        ndarray: Band power matrix with shape (n_channels, 5).
    '''
    bands = [
        (4, 8),    # theta
        (8, 12),   # alpha
        (12, 16),  # betaL
        (16, 25),  # betaH
        (25, 45)   # gamma
    ]

    freqs, psd = welch(
        segment,
        fs=sfreq,
        nperseg=segment.shape[1],
        axis=1
    )

    band_power = np.zeros((segment.shape[0], len(bands)))

    for band_i, (fmin, fmax) in enumerate(bands):
        idx = (freqs >= fmin) & (freqs < fmax)

        if np.sum(idx) == 0:
            band_power[:, band_i] = eps
        else:
            band_power[:, band_i] = np.trapezoid(
                psd[:, idx],
                freqs[idx],
                axis=1
            )

    return band_power


def epocx_band_ratio_hjorth_entropy_features(data, eps=1e-8):
    '''
    Computes EPOC X band power, Hjorth, entropy, and ratio features.

    Args:
        data: EEG data with shape (n_trials, n_secs, n_channels, sfreq).
        eps: Small value to avoid division by zero.

    Returns:
        ndarray: Feature matrix with shape
            (n_trials * n_secs, n_channels * 16).
    '''
    n_trials, n_secs, n_channels, sfreq = data.shape
    features_per_channel = 16

    features = np.empty(
        (n_trials, n_secs, n_channels * features_per_channel)
    )

    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            band_power = compute_band_power_welch(second, sfreq, eps)

            theta = band_power[:, 0]
            alpha = band_power[:, 1]
            beta_l = band_power[:, 2]
            beta_h = band_power[:, 3]
            gamma = band_power[:, 4]
            beta = beta_l + beta_h

            hjorth_mobility = mne_f.compute_hjorth_mobility_spect(
                sfreq,
                second
            )
            hjorth_complexity = mne_f.compute_hjorth_complexity_spect(
                sfreq,
                second
            )

            app_entropy = mne_f.compute_app_entropy(second)
            sample_entropy = mne_f.compute_samp_entropy(second)
            spectral_entropy = mne_f.compute_spect_entropy(sfreq, second)
            svd_entropy = mne_f.compute_svd_entropy(second)

            beta_alpha = beta / (alpha + eps)
            beta_alpha_theta = beta / (alpha + theta + eps)
            theta_beta = theta / (beta + eps)
            theta_alpha_beta = theta / (alpha + beta + eps)
            inverse_alpha = 1 / (alpha + eps)

            channel_features = np.column_stack([
                theta,
                alpha,
                beta_l,
                beta_h,
                gamma,
                hjorth_mobility,
                hjorth_complexity,
                app_entropy,
                sample_entropy,
                spectral_entropy,
                svd_entropy,
                beta_alpha,
                beta_alpha_theta,
                theta_beta,
                theta_alpha_beta,
                inverse_alpha
            ])

            features[i, j] = channel_features.reshape(-1)

    features = features.reshape(
        n_trials * n_secs,
        n_channels * features_per_channel
    )

    print("Input data shape:", data.shape)
    print("Output features shape:", features.shape)
    print("NaN count:", np.isnan(features).sum())
    print("Inf count:", np.isinf(features).sum())

    return features