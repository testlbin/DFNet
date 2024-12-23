import os
import csv
import tqdm
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc, logfbank, sigproc

def get_mean_fbank(path):
    # Read the WAV file
    (rate, sig) = wav.read(path)
    # Compute MFCC features
    mfcc_feat = mfcc(sig, rate)
    # Initialize list to hold averaged MFCC features
    mfcc_f = []
    # Compute average every 1 second
    leng = mfcc_feat.shape[0]
    leng = leng / 10 + 1
    leng = int(leng)

    for i in range(1, leng):
        mfcc_ = mfcc_feat[:i * 10, :]
        real_mfcc = np.mean(mfcc_, axis=0)
        mfcc_f.append(real_mfcc)
    mfcc_np = np.asarray(mfcc_f)

    return mfcc_np

def get_fbank(path):
    # Read the WAV file
    (rate, sig) = wav.read(path)
    # Compute filterbank features
    fbank_feat = logfbank(sig, rate)
    fbank_f = []
    # Compute average every 1 second
    leng = fbank_feat.shape[0]
    leng = leng / 100 + 1
    leng = int(leng)

    for i in range(1, leng):
        mfcc_ = fbank_feat[:i * 100, :]
        real_mfcc = np.mean(mfcc_, axis=0)
        fbank_f.append(real_mfcc)
    fbank_np = np.asarray(fbank_f)

    return fbank_np

def get_short_time_energy(path):
    # Read the WAV file
    (rate, sig) = wav.read(path)
    # Placeholder for actual implementation

def get_MFCC(dir_path, patient_name, video_path, mfcc_path):
    """
    Process directory path, patient name, video path, and path to save MFCC
    """
    if not os.path.exists(mfcc_path):
        os.makedirs(mfcc_path)
    if not os.path.exists(dir_path + 'label/'):
        os.makedirs(dir_path + 'label/')
    save_path = mfcc_path
    folder_path = video_path  # Replace with the actual path to your folder
    csv_data_path = dir_path + 'label/' + patient_name + '-label.csv' # Path for saving CSV file

    """
    Obtain paths for .wav files and corresponding labels
    """
    with open(csv_data_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File Name', 'Label'])

        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                label = filename.split('_')[0]
                label = 1 if label == 'snore' else 0
                writer.writerow([filename, label])

    print("Audio categorization complete!")

    """
    Automatically extract MFCC features for cut videos
    """
    print('Extracting MFCC features:', patient_name)
    data_list = []
    label_list = []

    with open(csv_data_path, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        rows = [row for row in reader]

    for patient in tqdm.tqdm(enumerate(rows), total=len(rows)):
        data_1 = patient[1]
        wav_path = data_1[0]
        label = data_1[1]

        get_mfcc_path = video_path + wav_path
        feature = get_mean_fbank(get_mfcc_path)

        data_list.append(feature)
        label_list.append(label)
    print("Feature extraction successful!")

    """
    Shuffle data and save as .npz file
    """
    data_array = np.array(data_list)
    label_array = np.array(label_list)
    shuffled_indices = np.random.permutation(len(data_array))
    shuffled_data = data_array[shuffled_indices]
    shuffled_labels = label_array[shuffled_indices]
    data_dict = {'data': shuffled_data, 'labels': shuffled_labels}
    np.savez(save_path + patient_name + '.npz', **data_dict)
    print("Data shuffling and saving complete!")
