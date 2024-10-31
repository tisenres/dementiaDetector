from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

# Define inputs
speech_input = Input(shape=speech_input_shape)
gait_input = Input(shape=gait_input_shape)
facial_input = Input(shape=facial_input_shape)

# Get features from each modality
speech_features = speech_model(speech_input)
gait_features = gait_model(gait_input)
facial_features = facial_model(facial_input)

# Fusion using Concatenation (simplified TFN)
from tensorflow.keras.layers import Multiply

# Create outer products (pairwise feature interactions)
speech_gait = Multiply()([speech_features, gait_features])
speech_facial = Multiply()([speech_features, facial_features])
gait_facial = Multiply()([gait_features, facial_features])

# Concatenate all features
fused_features = Concatenate()([speech_features, gait_features, facial_features,
                                speech_gait, speech_facial, gait_facial])

# Add fully connected layers after fusion
from tensorflow.keras.layers import Dense, Dropout

x = Dense(128, activation='relu')(fused_features)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

# Create the final model
tfn_model = Model(inputs=[speech_input, gait_input, facial_input], outputs=output)
tfn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
tfn_model.summary()