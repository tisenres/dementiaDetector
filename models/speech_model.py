from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

def build_speech_model(input_shape):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu')
    ])
    return model

# Example usage:
speech_input_shape = speech_features.shape[1:]
speech_model = build_speech_model(speech_input_shape)