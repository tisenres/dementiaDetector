from sklearn.model_selection import train_test_split

# Assuming labels are the same across modalities and properly aligned
X_speech_train, X_speech_test, y_train, y_test = train_test_split(
    speech_features, labels, test_size=0.2, random_state=42
)

X_gait_train, X_gait_test, _, _ = train_test_split(
    gait_features_padded, labels, test_size=0.2, random_state=42
)

X_facial_train, X_facial_test, _, _ = train_test_split(
    facial_images, labels, test_size=0.2, random_state=42
)

# Fit the model
tfn_model.fit(
    [X_speech_train, X_gait_train, X_facial_train],
    y_train,
    validation_data=([X_speech_test, X_gait_test, X_facial_test], y_test),
    epochs=30,
    batch_size=32
)

# Evaluate on the test set
loss, accuracy = tfn_model.evaluate([X_speech_test, X_gait_test, X_facial_test], y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Get predictions
y_pred = tfn_model.predict([X_speech_test, X_gait_test, X_facial_test])

# Convert probabilities to binary outputs
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate additional metrics
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred_binary))
print(confusion_matrix(y_test, y_pred_binary))