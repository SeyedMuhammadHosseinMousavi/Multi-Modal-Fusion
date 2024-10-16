%reset -f
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

# Reindex each DataFrame then they all have the same number of rows
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

# Combine all features into a single array for LSTM
# LSTM expects data in (samples, timesteps, features)
# We will treat each modality as a "time step" for simplicity
all_features = np.stack([ecg_standardized, eye_standardized, gsr_standardized], axis=1)

# One-hot encode labels for classification
n_classes = len(labels.unique())
y_encoded = to_categorical(labels, num_classes=n_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(all_features, y_encoded, test_size=0.2, random_state=42)

# Build an LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(all_features.shape[1], all_features.shape[2]), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
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

# Save the transformed data (LSTM input data) into a CSV file for later use
# Reshape back to 2D for saving
final_data_with_labels = pd.concat([pd.DataFrame(all_features.reshape(all_features.shape[0], -1)), labels.reset_index(drop=True)], axis=1)
final_data_with_labels.to_csv('LSTM_input_data_with_labels.csv', index=False)

print("Data saved to 'LSTM_input_data_with_labels.csv'.")
