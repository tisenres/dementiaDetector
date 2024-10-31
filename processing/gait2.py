from tensorflow.keras.preprocessing.sequence import pad_sequences

max_timesteps = 100  # Define based on your data distribution

gait_features_padded = pad_sequences(
    gait_features_list, maxlen=max_timesteps, dtype='float32', padding='post', truncating='post'
)