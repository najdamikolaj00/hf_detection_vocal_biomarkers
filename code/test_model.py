"""
This script demonstrates how to load audio, extract features using OpenSMILE, and make predictions using a trained model.
Replace the placeholder strings with the actual paths to your audio file, selected features file, and trained model file.

Expected structure of the feature file (features_for_model):
- The selected features file is expected to be a CSV file with a column named 'Feature Names'
  that contains the feature names used in the model.

Expected structure of the feature file (features_for_model):
- The selected features file is expected to be a CSV file with a column named 'Feature Names'
  that contains the feature names used in the model.
"""
import torch
import torchaudio
import opensmile
import pandas as pd
import pickle

def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    resampled_audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    rms = torch.sqrt(torch.mean(resampled_audio**2))
    resampled_audio = resampled_audio / rms

    return resampled_audio

def extract_features(resampled_audio):
    feature_entry = {}
    feature_levels = [
        opensmile.FeatureLevel.LowLevelDescriptors,
        opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
        opensmile.FeatureLevel.Functionals
    ]

    for level in feature_levels:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=level,
            channels=[0]
        )
        features = smile.process_signal(resampled_audio, 16000)
        feature_names = smile.feature_names
        feature_means = features.mean(axis=0)

        for feature_name, mean_value in zip(feature_names, feature_means):
            feature_entry[f"{feature_name}"] = mean_value

    return feature_entry

def predict_result(test_features, features_for_model, model_filename):
    # Load selected features and scale
    selected_feature_names_df = pd.read_csv(features_for_model)
    selected_feature_names = selected_feature_names_df['Feature Names'].tolist()

    test_features = test_features.T[selected_feature_names]
    test_tensor = torch.tensor(test_features.values, dtype=torch.float32).cpu()
    max_value = torch.max(test_tensor)

    # Scale the tensor between 0 and 1
    scaled_tensor = torch.div(test_tensor, max_value)

    # Load the trained model and make predictions
    loaded_model = pickle.load(open(model_filename, 'rb'))
    result = loaded_model.predict(scaled_tensor)
    return result

if __name__ == "__main__":
    # Placeholder paths - replace these with your actual paths
    audio_path = 'path/to/your/audio/file.wav'
    features_for_model = 'path/to/your/selected/features.csv'
    model_filename = 'path/to/your/trained/model.pickle'

    # Load and process audio, extract features, and make predictions
    resampled_audio = load_audio(audio_path)
    feature_entry = extract_features(resampled_audio)
    feature_entry_df = pd.DataFrame.from_dict(feature_entry, orient='index')

    # Make prediction
    result = predict_result(feature_entry_df, features_for_model, model_filename)
    print("Prediction: ", result)  # 1 = HF
