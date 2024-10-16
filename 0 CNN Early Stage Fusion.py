%reset -f
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
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

# Imputation to handle NaNs in features
imputer = KNNImputer(n_neighbors=5)
ecg_features_imputed = pd.DataFrame(imputer.fit_transform(ecg_features), columns=ecg_features.columns)
eye_tracking_features_imputed = pd.DataFrame(imputer.fit_transform(eye_tracking_features), columns=eye_tracking_features.columns)
gsr_features_imputed = pd.DataFrame(imputer.fit_transform(gsr_features), columns=gsr_features.columns)

# Standardize the features
scaler = StandardScaler()
ecg_standardized = scaler.fit_transform(ecg_features_imputed)
eye_standardized = scaler.fit_transform(eye_tracking_features_imputed)
gsr_standardized = scaler.fit_transform(gsr_features_imputed)

# Combine all features into a single array
all_features = np.hstack((ecg_standardized, eye_standardized, gsr_standardized))

# Reshape features for CNN (CNN requires 3D input: samples, width, height, channels)
n_samples = all_features.shape[0]
n_features = all_features.shape[1]

# Reshape into 3D for CNN: (samples, width, height, channels)
# Here, we treat each sample as a "1D image" with 1 channel for each modality
cnn_input = all_features.reshape(n_samples, n_features, 1, 1)  # Reshape to 3D

# One-hot encode labels for classification
n_classes = len(labels.unique())
y_encoded = to_categorical(labels, num_classes=n_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(cnn_input, y_encoded, test_size=0.2, random_state=42)

# Build a CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 1), activation='relu', input_shape=(n_features, 1, 1)))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

# Compile the CNN model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

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

# Save the transformed data (CNN input data) into a CSV file for later use
# We flatten the CNN input back into 2D for saving
cnn_input_flattened = cnn_input.reshape(n_samples, n_features)
final_data_with_labels = pd.concat([pd.DataFrame(cnn_input_flattened), labels.reset_index(drop=True)], axis=1)
final_data_with_labels.to_csv('CNN_input_data_with_labels.csv', index=False)

print("Data saved to 'CNN_input_data_with_labels.csv'.")
