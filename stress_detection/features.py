from scipy.signal import welch
import numpy as np
import mne_features.univariate as mne_f
from scipy.signal import butter, sosfiltfilt, welch

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
        [n_trials * n_secs, n_channels * features_per_channel])
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
    features_per_channel = len(freq_bands) - 1

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            psd = mne_f.compute_pow_freq_bands(
                sfreq, second, freq_bands=freq_bands)
            features[i][j] = psd
    features = features.reshape(
        [n_trials * n_secs, n_channels * features_per_channel])
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
        [n_trials * n_secs, n_channels * features_per_channel])
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
        [n_trials * n_secs, n_channels * features_per_channel])
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
        [n_trials * n_secs, n_channels * features_per_channel])
    return features


def bandpass_filter_segment(segment, sfreq, fmin, fmax):
    '''
    Bandpass filters one EEG segment.

    Args:
        segment: EEG segment with shape (n_channels, n_samples).
        sfreq: Sampling frequency.
        fmin: Lower frequency.
        fmax: Upper frequency.

    Returns:
        ndarray: Band-filtered EEG segment with shape (n_channels, n_samples).
    '''
    sos = butter(
        N=4,
        Wn=[fmin, fmax],
        btype="bandpass",
        fs=sfreq,
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        segment,
        axis=1
    )

    return filtered


def compute_band_power_from_signal(segment, sfreq):
    '''
    Computes power for one already band-filtered EEG signal.

    Args:
        segment: Band-filtered EEG segment with shape (n_channels, n_samples).
        sfreq: Sampling frequency.

    Returns:
        ndarray: Power value for each channel.
    '''
    freqs, psd = welch(
        segment,
        fs=sfreq,
        nperseg=segment.shape[1],
        axis=1
    )

    power = np.trapezoid(
        psd,
        freqs,
        axis=1
    )

    return power


def epocx_bandwise_hjorth_entropy_features(data, eps=1e-8):
    '''
    Computes EPOC X aligned band-wise features.

    For each channel and each frequency band, this function computes:
        band_power
        hjorth_activity
        hjorth_mobility
        hjorth_complexity
        app_entropy
        sample_entropy
        spectral_entropy
        svd_entropy

    It also computes ratio features per channel:
        beta / alpha
        beta / (alpha + theta)
        theta / beta
        theta / (alpha + beta)
        1 / alpha

    Args:
        data: EEG data with shape (n_trials, n_secs, n_channels, sfreq).
        eps: Small value to avoid division by zero.

    Returns:
        ndarray: Feature matrix with shape
            (n_trials * n_secs, n_channels * 45).
    '''
    n_trials, n_secs, n_channels, sfreq = data.shape

    bands = {
        "theta": (4, 8),
        "alpha": (8, 12),
        "betaL": (12, 16),
        "betaH": (16, 25),
        "gamma": (25, 45)
    }

    band_feature_count = 8
    ratio_feature_count = 5
    features_per_channel = len(bands) * band_feature_count + ratio_feature_count

    features = np.empty(
        (n_trials, n_secs, n_channels * features_per_channel)
    )

    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            band_features_by_name = {}
            band_powers_by_name = {}

            for band_name, (fmin, fmax) in bands.items():
                band_signal = bandpass_filter_segment(
                    second,
                    sfreq,
                    fmin,
                    fmax
                )

                band_power = compute_band_power_from_signal(
                    band_signal,
                    sfreq
                )

                hjorth_activity = mne_f.compute_variance(band_signal)

                hjorth_mobility = mne_f.compute_hjorth_mobility_spect(
                    sfreq,
                    band_signal
                )

                hjorth_complexity = mne_f.compute_hjorth_complexity_spect(
                    sfreq,
                    band_signal
                )

                app_entropy = mne_f.compute_app_entropy(band_signal)
                sample_entropy = mne_f.compute_samp_entropy(band_signal)

                spectral_entropy = mne_f.compute_spect_entropy(
                    sfreq,
                    band_signal
                )

                svd_entropy = mne_f.compute_svd_entropy(band_signal)

                band_features = np.column_stack([
                    band_power,
                    hjorth_activity,
                    hjorth_mobility,
                    hjorth_complexity,
                    app_entropy,
                    sample_entropy,
                    spectral_entropy,
                    svd_entropy
                ])

                band_features_by_name[band_name] = band_features
                band_powers_by_name[band_name] = band_power

            theta = band_powers_by_name["theta"]
            alpha = band_powers_by_name["alpha"]
            beta_l = band_powers_by_name["betaL"]
            beta_h = band_powers_by_name["betaH"]
            beta = beta_l + beta_h

            beta_alpha = beta / (alpha + eps)
            beta_alpha_theta = beta / (alpha + theta + eps)
            theta_beta = theta / (beta + eps)
            theta_alpha_beta = theta / (alpha + beta + eps)
            inverse_alpha = 1 / (alpha + eps)

            ratio_features = np.column_stack([
                beta_alpha,
                beta_alpha_theta,
                theta_beta,
                theta_alpha_beta,
                inverse_alpha
            ])

            channel_features = []

            for ch in range(n_channels):
                one_channel_features = []

                for band_name in bands.keys():
                    one_channel_features.extend(
                        band_features_by_name[band_name][ch]
                    )

                one_channel_features.extend(ratio_features[ch])

                channel_features.extend(one_channel_features)

            features[i, j] = np.array(channel_features)

    features = features.reshape(
        n_trials * n_secs,
        n_channels * features_per_channel
    )

    print("Input data shape:", data.shape)
    print("Output features shape:", features.shape)
    print("Features per channel:", features_per_channel)
    print("NaN count:", np.isnan(features).sum())
    print("Inf count:", np.isinf(features).sum())

    return features
