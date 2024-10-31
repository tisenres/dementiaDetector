import librosa
import numpy as np
import os

def extract_mfcc_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)  # Load audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Take the mean over time
    return mfcc_mean

# Example usage:
speech_features = []
labels = []

speech_data_dir = 'path_to_speech_data'  # Replace with your path

for filename in os.listdir(speech_data_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(speech_data_dir, filename)
        features = extract_mfcc_features(file_path)
        speech_features.append(features)
        # Extract label from filename or separate metadata
        label = 1 if 'dementia' in filename else 0
        labels.append(label)

speech_features = np.array(speech_features)
labels = np.array(labels)