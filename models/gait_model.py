from tensorflow.keras.layers import LSTM, Dropout

def build_gait_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(32, activation='relu')
    ])
    return model

# Example usage:
gait_input_shape = (max_timesteps, 3)  # Assuming 3 features: acc_x, acc_y, acc_z
gait_model = build_gait_model(gait_input_shape)