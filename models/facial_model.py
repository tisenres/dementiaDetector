from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

def build_facial_model(input_shape):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu')
    ])
    return model

# Example usage:
facial_input_shape = (48, 48, 1)
facial_model = build_facial_model(facial_input_shape)