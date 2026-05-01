import os
import numpy as np
import pandas as pd
import scipy
import variables as v


def load_dataset(data_type="ica_filtered", test_type="Arithmetic"):
    '''
    Loads data from the SAM 40 Dataset and keeps selected channels.

    Args:
        data_type (string): The data type to load. Defaults to "ica_filtered".
        test_type (string): The test type to load. Defaults to "Arithmetic".

    Returns:
        ndarray: The specified dataset with selected channels only.
    '''
    assert test_type in v.TEST_TYPES
    assert data_type in v.DATA_TYPES

    if data_type == "ica_filtered" and test_type != "Arithmetic":
        print("Data of type", data_type, "does not have test type", test_type)
        return 0

    if data_type == "raw":
        dir = v.DIR_RAW
        data_key = "Data"
    elif data_type == "wt_filtered":
        dir = v.DIR_FILTERED
        data_key = "Clean_data"
    else:
        dir = v.DIR_ICA_FILTERED
        data_key = "Clean_data"

    selected_channel_numbers = [3, 4, 5, 8, 10, 16, 19, 25, 27, 30, 31, 32]
    selected_channel_names = ["Fp1", "F7", "F3", "FC5", "T7", "O1", "O2", "T8", "FC6", "F4", "F8", "Fp2"]
    epocx_names = ["POW.AF3", "POW.F7", "POW.F3", "POW.FC5", "POW.T7", "POW.O1",
                   "POW.O2", "POW.T8", "POW.FC6", "POW.F4", "POW.F8", "POW.AF4"]

    selected_channel_indices = [ch - 1 for ch in selected_channel_numbers]

    dataset = np.empty((120, len(selected_channel_indices), 3200))

    print("Selected channel numbers:", selected_channel_numbers)
    print("Selected SAM40 channel names:", selected_channel_names)
    print("Mapped EPOC X channel names:", epocx_names)
    print("Selected Python indices:", selected_channel_indices)
    print("Expected dataset shape:", dataset.shape)

    counter = 0
    for filename in os.listdir(dir):
        if test_type not in filename:
            continue

        f = os.path.join(dir, filename)
        data = scipy.io.loadmat(f)[data_key]

        if counter == 0:
            print("First loaded file:", filename)
            print("Original data shape:", data.shape)
            print("Original channel count:", data.shape[0])
            print("Original sample count:", data.shape[1])

        data = data[selected_channel_indices, :]

        if counter == 0:
            print("Selected data shape:", data.shape)
            print("Selected channels:")
            for number, sam_name, epocx_name, index in zip(
                    selected_channel_numbers,
                    selected_channel_names,
                    epocx_names,
                    selected_channel_indices
            ):
                print(
                    f"  original number {number}, Python index {index}, "
                    f"SAM40 name {sam_name}, EPOC X name {epocx_name}"
                )

        dataset[counter] = data
        counter += 1

    print("Loaded trials:", counter)
    print("Final dataset shape:", dataset.shape)

    return dataset


def load_labels():
    '''
    Loads labels from the dataset and transforms stress scale values
    into three stress level labels.

    Returns:
        DataFrame: Labels with subject_no unchanged and trial columns classified.
    '''
    labels = pd.read_excel(v.LABELS_PATH)
    labels = labels.rename(columns=v.COLUMNS_TO_RENAME)
    labels = labels[1:].copy()

    label_columns = [col for col in labels.columns if col != "subject_no"]

    labels["subject_no"] = labels["subject_no"].astype("int")
    labels[label_columns] = labels[label_columns].astype("int")

    def classify_stress(value):
        if 1 <= value <= 4:
            return "mild_stress"
        if 5 <= value <= 7:
            return "stress"
        if 8 <= value <= 10:
            return "high_stress"
        return None

    labels[label_columns] = labels[label_columns].map(classify_stress)

    return labels


def format_labels(labels, test_type="Arithmetic", epochs=1):
    '''
    Filter the labels and repeat for the specified amount of epochs.

    Args:
        labels (ndarray): The labels.
        test_type (string): The test_type to filter by. Defaults to "Arithmetic".
        epochs (int): The amount of epochs. Defaults to 1.

    Returns:
        ndarray: The formatted labels.

    '''
    assert (test_type in v.TEST_TYPES)

    formatted_labels = []
    for trial in v.TEST_TYPE_COLUMNS[test_type]:
        formatted_labels.append(labels[trial])

    formatted_labels = pd.concat(formatted_labels).to_numpy()

    formatted_labels = formatted_labels.repeat(epochs)

    return formatted_labels


def split_data(data, sfreq):
    '''
    Splits EEG data into epochs with length 1 sec.

    Args:
        data (ndarray): EEG data.
        sfreq (int): The sampling frequency.
    
    Returns:
        ndarray: The epoched data.

    '''

    n_trials, n_channels, n_samples = data.shape

    epoched_data = np.empty((n_trials, n_samples // sfreq, n_channels, sfreq))
    for i in range(data.shape[0]):
        for j in range(data.shape[2] // sfreq):
            epoched_data[i, j] = data[i, :, j * sfreq:(j + 1) * sfreq]
    return epoched_data
