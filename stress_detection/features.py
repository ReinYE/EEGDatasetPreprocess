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


import numpy as np
from scipy.signal import welch


def compute_band_power_welch(segment, sfreq):
    '''
    Computes EPOC X aligned band power from one EEG epoch.

    Args:
        segment: EEG segment with shape (n_channels, n_samples).
        sfreq: Sampling frequency.

    Returns:
        ndarray: Band power matrix with shape (n_channels, 5).
    '''
    bands = [
        (4, 8),     # theta
        (8, 12),    # alpha
        (12, 16),   # betaL
        (16, 25),   # betaH
        (25, 45)    # gamma
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
            band_power[:, band_i] = 0
        else:
            band_power[:, band_i] = np.trapezoid(
                psd[:, idx],
                freqs[idx],
                axis=1
            )

    return band_power


def hjorth_from_power_sequence(x, eps=1e-8):
    '''
    Computes Hjorth mobility and complexity from a band power sequence.

    Args:
        x: One-dimensional band power sequence.
        eps: Small value to avoid division by zero.

    Returns:
        tuple: mobility, complexity.
    '''
    x = np.asarray(x, dtype=float)

    if len(x) < 3:
        return np.nan, np.nan

    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    mobility = np.sqrt(var_dx / (var_x + eps))

    mobility_dx = np.sqrt(var_ddx / (var_dx + eps))

    complexity = mobility_dx / (mobility + eps)

    return mobility, complexity


def epocx_bandpower_hjorth_features(data, power_window=None, step=1, eps=1e-8):
    '''
    Computes Hjorth mobility and complexity on band power sequences.

    This function first computes EPOC X aligned band power from raw EEG,
    then computes Hjorth mobility and complexity from the band power sequence.

    Args:
        data: EEG data with shape (n_trials, n_secs, n_channels, sfreq).
        power_window: Number of band power samples used to compute Hjorth.
            If None, use the whole trial.
            If an integer, use rolling windows.
        step: Step size for rolling windows.
        eps: Small value to avoid division by zero.

    Returns:
        ndarray: Feature matrix.
            If power_window is None:
                shape = (n_trials, n_channels * 15)
            If power_window is int:
                shape = (n_trials * n_windows, n_channels * 15)
    '''
    n_trials, n_secs, n_channels, sfreq = data.shape

    band_names = ["theta", "alpha", "betaL", "betaH", "gamma"]
    n_bands = len(band_names)

    features_per_channel = n_bands * 2 + 5

    # ========= 1. Compute band power sequence =========
    band_power_seq = np.empty((n_trials, n_secs, n_channels, n_bands))

    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            band_power_seq[i, j] = compute_band_power_welch(second, sfreq)

    print("Band power sequence shape:", band_power_seq.shape)
    print("Meaning: n_trials, n_secs, n_channels, n_bands")

    # ========= 2. Decide Hjorth window =========
    if power_window is None:
        power_window = n_secs
        use_full_trial = True
    else:
        use_full_trial = False

    if power_window < 3:
        raise ValueError("power_window must be at least 3 to compute Hjorth complexity.")

    if power_window > n_secs:
        raise ValueError(
            f"power_window={power_window} is larger than n_secs={n_secs}."
        )

    n_windows = (n_secs - power_window) // step + 1

    all_features = []

    # ========= 3. Compute Hjorth on band power sequence =========
    for trial_i in range(n_trials):
        for start in range(0, n_secs - power_window + 1, step):
            end = start + power_window

            window_power = band_power_seq[trial_i, start:end]

            # shape: (power_window, n_channels, 5)

            channel_features = []

            for ch in range(n_channels):
                one_channel_features = []

                for band_i in range(n_bands):
                    power_series = window_power[:, ch, band_i]

                    mobility, complexity = hjorth_from_power_sequence(
                        power_series,
                        eps=eps
                    )

                    one_channel_features.extend([
                        mobility,
                        complexity
                    ])

                # ========= 4. Ratio features from average band power in this window =========
                mean_power = np.mean(window_power[:, ch, :], axis=0)

                theta = mean_power[0]
                alpha = mean_power[1]
                beta_l = mean_power[2]
                beta_h = mean_power[3]
                beta = beta_l + beta_h

                beta_alpha = beta / (alpha + eps)
                beta_alpha_theta = beta / (alpha + theta + eps)
                theta_beta = theta / (beta + eps)
                theta_alpha_beta = theta / (alpha + beta + eps)
                inverse_alpha = 1 / (alpha + eps)

                one_channel_features.extend([
                    beta_alpha,
                    beta_alpha_theta,
                    theta_beta,
                    theta_alpha_beta,
                    inverse_alpha
                ])

                channel_features.extend(one_channel_features)

            all_features.append(channel_features)

            if use_full_trial:
                break

    features = np.asarray(all_features)

    print("Input EEG data shape:", data.shape)
    print("Output features shape:", features.shape)
    print("Features per channel:", features_per_channel)
    print("NaN count:", np.isnan(features).sum())
    print("Inf count:", np.isinf(features).sum())

    return features

def epocx_relative_power_hjorth_features(data, power_window=None, step=1, eps=1e-8):
    '''
    Computes Hjorth mobility and complexity on relative band power sequences.

    This function first computes EPOC X aligned absolute band power from raw EEG,
    then converts it to relative band power, and finally computes Hjorth mobility
    and Hjorth complexity from the relative power sequence.

    For each channel, this function computes:
        theta_relative_power_hjorth_mobility
        theta_relative_power_hjorth_complexity
        alpha_relative_power_hjorth_mobility
        alpha_relative_power_hjorth_complexity
        betaL_relative_power_hjorth_mobility
        betaL_relative_power_hjorth_complexity
        betaH_relative_power_hjorth_mobility
        betaH_relative_power_hjorth_complexity
        gamma_relative_power_hjorth_mobility
        gamma_relative_power_hjorth_complexity

    It also computes ratio features per channel from mean relative power:
        beta / alpha
        beta / (alpha + theta)
        theta / beta
        theta / (alpha + beta)
        1 / alpha

    Args:
        data: EEG data with shape (n_trials, n_secs, n_channels, sfreq).
        power_window: Number of relative power samples used to compute Hjorth.
            If None, use the whole trial.
            If an integer, use rolling windows.
        step: Step size for rolling windows.
        eps: Small value to avoid division by zero.

    Returns:
        ndarray: Feature matrix.
            If power_window is None:
                shape = (n_trials, n_channels * 15)
            If power_window is int:
                shape = (n_trials * n_windows, n_channels * 15)
    '''
    n_trials, n_secs, n_channels, sfreq = data.shape

    band_names = ["theta", "alpha", "betaL", "betaH", "gamma"]
    n_bands = len(band_names)

    features_per_channel = n_bands * 2 + 5

    # ========= 1. Compute absolute band power sequence =========
    abs_power_seq = np.empty((n_trials, n_secs, n_channels, n_bands))

    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            abs_power_seq[i, j] = compute_band_power_welch(second, sfreq)

    # ========= 2. Convert absolute power to relative power =========
    total_power = np.sum(abs_power_seq, axis=3, keepdims=True)
    relative_power_seq = abs_power_seq / (total_power + eps)

    print("Absolute band power sequence shape:", abs_power_seq.shape)
    print("Relative band power sequence shape:", relative_power_seq.shape)
    print("Meaning: n_trials, n_secs, n_channels, n_bands")

    # ========= 3. Decide Hjorth window =========
    if power_window is None:
        power_window = n_secs
        use_full_trial = True
    else:
        use_full_trial = False

    if power_window < 3:
        raise ValueError("power_window must be at least 3 to compute Hjorth complexity.")

    if power_window > n_secs:
        raise ValueError(
            f"power_window={power_window} is larger than n_secs={n_secs}."
        )

    n_windows = (n_secs - power_window) // step + 1

    all_features = []

    # ========= 4. Compute Hjorth on relative power sequence =========
    for trial_i in range(n_trials):
        for start in range(0, n_secs - power_window + 1, step):
            end = start + power_window

            window_relative_power = relative_power_seq[trial_i, start:end]
            # shape: (power_window, n_channels, 5)

            channel_features = []

            for ch in range(n_channels):
                one_channel_features = []

                for band_i in range(n_bands):
                    relative_power_series = window_relative_power[:, ch, band_i]

                    mobility, complexity = hjorth_from_power_sequence(
                        relative_power_series,
                        eps=eps
                    )

                    one_channel_features.extend([
                        mobility,
                        complexity
                    ])

                # ========= 5. Ratio features from mean relative power =========
                mean_relative_power = np.mean(
                    window_relative_power[:, ch, :],
                    axis=0
                )

                theta = mean_relative_power[0]
                alpha = mean_relative_power[1]
                beta_l = mean_relative_power[2]
                beta_h = mean_relative_power[3]
                beta = beta_l + beta_h

                beta_alpha = beta / (alpha + eps)
                beta_alpha_theta = beta / (alpha + theta + eps)
                theta_beta = theta / (beta + eps)
                theta_alpha_beta = theta / (alpha + beta + eps)
                inverse_alpha = 1 / (alpha + eps)

                one_channel_features.extend([
                    beta_alpha,
                    beta_alpha_theta,
                    theta_beta,
                    theta_alpha_beta,
                    inverse_alpha
                ])

                channel_features.extend(one_channel_features)

            all_features.append(channel_features)

            if use_full_trial:
                break

    features = np.asarray(all_features)

    print("Input EEG data shape:", data.shape)
    print("Output relative-power Hjorth features shape:", features.shape)
    print("Features per channel:", features_per_channel)
    print("Power window:", power_window)
    print("Step:", step)
    print("Number of windows per trial:", n_windows)
    print("NaN count:", np.isnan(features).sum())
    print("Inf count:", np.isinf(features).sum())

    return features


def epocx_relative_power_hjorth_features_plus(data, power_window=None, step=1, eps=1e-8):
    '''
    Computes Hjorth mobility and complexity on relative band power sequences
    and relative-power ratio sequences.

    This function first computes EPOC X aligned absolute band power from raw EEG,
    then converts it to relative band power, and finally computes Hjorth mobility
    and Hjorth complexity from:
        1. each relative band power sequence
        2. each ratio sequence

    For each channel, this function computes:

        Relative power Hjorth features:
            theta_relative_power_hjorth_mobility
            theta_relative_power_hjorth_complexity
            alpha_relative_power_hjorth_mobility
            alpha_relative_power_hjorth_complexity
            betaL_relative_power_hjorth_mobility
            betaL_relative_power_hjorth_complexity
            betaH_relative_power_hjorth_mobility
            betaH_relative_power_hjorth_complexity
            gamma_relative_power_hjorth_mobility
            gamma_relative_power_hjorth_complexity

        Ratio Hjorth features:
            beta_alpha_hjorth_mobility
            beta_alpha_hjorth_complexity
            beta_alpha_theta_hjorth_mobility
            beta_alpha_theta_hjorth_complexity
            theta_beta_hjorth_mobility
            theta_beta_hjorth_complexity
            theta_alpha_beta_hjorth_mobility
            theta_alpha_beta_hjorth_complexity
            inverse_alpha_hjorth_mobility
            inverse_alpha_hjorth_complexity

    Args:
        data: EEG data with shape (n_trials, n_secs, n_channels, sfreq).
        power_window: Number of relative power samples used to compute Hjorth.
            If None, use the whole trial.
            If an integer, use rolling windows.
        step: Step size for rolling windows.
        eps: Small value to avoid division by zero.

    Returns:
        ndarray: Feature matrix.
            If power_window is None:
                shape = (n_trials, n_channels * 20)
            If power_window is int:
                shape = (n_trials * n_windows, n_channels * 20)
    '''
    n_trials, n_secs, n_channels, sfreq = data.shape

    band_names = ["theta", "alpha", "betaL", "betaH", "gamma"]
    n_bands = len(band_names)

    ratio_names = [
        "beta_alpha",
        "beta_alpha_theta",
        "theta_beta",
        "theta_alpha_beta",
        "inverse_alpha"
    ]

    features_per_channel = (n_bands * 2) + (len(ratio_names) * 2)

    # ========= 1. Compute absolute band power sequence =========
    abs_power_seq = np.empty((n_trials, n_secs, n_channels, n_bands))

    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            abs_power_seq[i, j] = compute_band_power_welch(second, sfreq)

    # ========= 2. Convert absolute power to relative power =========
    total_power = np.sum(abs_power_seq, axis=3, keepdims=True)
    relative_power_seq = abs_power_seq / (total_power + eps)

    print("Absolute band power sequence shape:", abs_power_seq.shape)
    print("Relative band power sequence shape:", relative_power_seq.shape)
    print("Meaning: n_trials, n_secs, n_channels, n_bands")

    # ========= 3. Decide Hjorth window =========
    if power_window is None:
        power_window = n_secs
        use_full_trial = True
    else:
        use_full_trial = False

    if power_window < 3:
        raise ValueError("power_window must be at least 3 to compute Hjorth complexity.")

    if power_window > n_secs:
        raise ValueError(
            f"power_window={power_window} is larger than n_secs={n_secs}."
        )

    n_windows = (n_secs - power_window) // step + 1

    all_features = []

    # ========= 4. Compute Hjorth on relative power and ratio sequences =========
    for trial_i in range(n_trials):
        for start in range(0, n_secs - power_window + 1, step):
            end = start + power_window

            window_relative_power = relative_power_seq[trial_i, start:end]
            # shape: (power_window, n_channels, 5)

            channel_features = []

            for ch in range(n_channels):
                one_channel_features = []

                # ===== 4.1 Hjorth on each relative band power sequence =====
                for band_i in range(n_bands):
                    relative_power_series = window_relative_power[:, ch, band_i]

                    mobility, complexity = hjorth_from_power_sequence(
                        relative_power_series,
                        eps=eps
                    )

                    one_channel_features.extend([
                        mobility,
                        complexity
                    ])

                # ===== 4.2 Build ratio sequences from relative power sequences =====
                theta_series = window_relative_power[:, ch, 0]
                alpha_series = window_relative_power[:, ch, 1]
                beta_l_series = window_relative_power[:, ch, 2]
                beta_h_series = window_relative_power[:, ch, 3]
                beta_series = beta_l_series + beta_h_series

                ratio_series_list = [
                    beta_series / (alpha_series + eps),
                    beta_series / (alpha_series + theta_series + eps),
                    theta_series / (beta_series + eps),
                    theta_series / (alpha_series + beta_series + eps),
                    1 / (alpha_series + eps)
                ]

                # ===== 4.3 Hjorth on each ratio sequence =====
                for ratio_series in ratio_series_list:
                    mobility, complexity = hjorth_from_power_sequence(
                        ratio_series,
                        eps=eps
                    )

                    one_channel_features.extend([
                        mobility,
                        complexity
                    ])

                channel_features.extend(one_channel_features)

            all_features.append(channel_features)

            if use_full_trial:
                break

    features = np.asarray(all_features)

    print("Input EEG data shape:", data.shape)
    print("Output relative-power Hjorth features shape:", features.shape)
    print("Features per channel:", features_per_channel)
    print("Power window:", power_window)
    print("Step:", step)
    print("Number of windows per trial:", n_windows)
    print("NaN count:", np.isnan(features).sum())
    print("Inf count:", np.isinf(features).sum())

    return features