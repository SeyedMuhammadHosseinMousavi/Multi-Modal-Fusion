%reset -f
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# Load the CSV files into pandas DataFrames
ecg_df = pd.read_csv('ECG.csv')
eye_tracking_df = pd.read_csv('EyeTracking.csv')
gsr_df = pd.read_csv('GSR.csv')

# the first column is labeled consistently as 'Label'
ecg_df.columns = ['Label'] + list(ecg_df.columns[1:])
eye_tracking_df.columns = ['Label'] + list(eye_tracking_df.columns[1:])
gsr_df.columns = ['Label'] + list(gsr_df.columns[1:])

# Padding DataFrames to the same number of rows (using NaN where data is missing)
max_rows = max(len(ecg_df), len(eye_tracking_df), len(gsr_df))

# Reindex each DataFrame to ensure they all have the same number of rows
ecg_df = ecg_df.reindex(range(max_rows), fill_value=np.nan)
eye_tracking_df = eye_tracking_df.reindex(range(max_rows), fill_value=np.nan)
gsr_df = gsr_df.reindex(range(max_rows), fill_value=np.nan)

# Separate features (dropping the 'Label' column)
ecg_features = ecg_df.drop(columns=['Label'])
eye_tracking_features = eye_tracking_df.drop(columns=['Label'])
gsr_features = gsr_df.drop(columns=['Label'])

# Handle missing labels by forward-filling
labels = ecg_df['Label'].fillna(method='ffill')

# KNN Imputation for missing values in the features
imputer = KNNImputer(n_neighbors=5)
ecg_features_imputed = pd.DataFrame(imputer.fit_transform(ecg_features), columns=ecg_features.columns)
eye_tracking_features_imputed = pd.DataFrame(imputer.fit_transform(eye_tracking_features), columns=eye_tracking_features.columns)
gsr_features_imputed = pd.DataFrame(imputer.fit_transform(gsr_features), columns=gsr_features.columns)

# Get the maximum number of columns (features) across all modalities
max_features = max(ecg_features_imputed.shape[1], eye_tracking_features_imputed.shape[1], gsr_features_imputed.shape[1])

# Pad each modality with zeros to have the same number of features (columns)
ecg_features_padded = np.pad(ecg_features_imputed, ((0, 0), (0, max_features - ecg_features_imputed.shape[1])), 'constant')
eye_tracking_features_padded = np.pad(eye_tracking_features_imputed, ((0, 0), (0, max_features - eye_tracking_features_imputed.shape[1])), 'constant')
gsr_features_padded = np.pad(gsr_features_imputed, ((0, 0), (0, max_features - gsr_features_imputed.shape[1])), 'constant')

# Standardize the features after padding
scaler = StandardScaler()
ecg_standardized = scaler.fit_transform(ecg_features_padded)
eye_standardized = scaler.fit_transform(eye_tracking_features_padded)
gsr_standardized = scaler.fit_transform(gsr_features_padded)

# Combine all features into a single array for the Autoencoder
all_features = np.hstack((ecg_standardized, eye_standardized, gsr_standardized))

# Autoencoder architecture
input_dim = all_features.shape[1]

# Define encoder
encoding_dim = 64  # Size of the encoded representation
input_layer = Input(shape=(input_dim,))
encoder = Dense(128, activation='relu')(input_layer)
encoder = Dense(encoding_dim, activation='relu')(encoder)

# Define decoder
decoder = Dense(128, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

# Full autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(all_features, all_features, epochs=200, batch_size=32, validation_split=0.2)

# Extract the encoder part for feature extraction
encoder_model = Model(inputs=input_layer, outputs=encoder)
encoded_features = encoder_model.predict(all_features)

# One-hot encode labels for classification
n_classes = len(labels.unique())
y_encoded = to_categorical(labels, num_classes=n_classes)

# Train-test split on encoded features
X_train, X_test, y_train, y_test = train_test_split(encoded_features, y_encoded, test_size=0.2, random_state=42)

# Build a simple classifier using the encoded features
classifier = Sequential()
classifier.add(Dense(64, activation='relu', input_shape=(encoding_dim,)))
classifier.add(Dense(n_classes, activation='softmax'))

# Compile the classifier
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the classifier
classifier.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the classifier
y_pred = classifier.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print the classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# Print the final test accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Save the encoded data (Autoencoder output) into a CSV file for later use
encoded_data_with_labels = pd.concat([pd.DataFrame(encoded_features), labels.reset_index(drop=True)], axis=1)
encoded_data_with_labels.to_csv('Autoencoder_encoded_data_with_labels.csv', index=False)

print("Encoded data saved to 'Autoencoder_encoded_data_with_labels.csv'.")
